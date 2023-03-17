# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import copy

import paddle
import paddle.distributed as dist
from paddle.optimizer.lr import LRScheduler
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../../../')))

from ppfleetx.utils.log import logger
from ppfleetx.distributed.apis import env
import ppfleetx.models.language_model.gpt as gpt
from ppfleetx.utils.tensor_fusion_helper import all_reduce_parameters
from ppfleetx.data.tokenizers import GPTTokenizer, GPTChineseTokenizer
from ppfleetx.models.language_model.gpt.dygraph.sequence_parallel_utils import register_sequence_parallel_allreduce_hooks

MODEL_CLASSES = {
    "GPT": (GPTTokenizer, "gpt2"),
    "GPT-cn": (GPTChineseTokenizer, "gpt-cpm-large-cn"),
}


def _get_model_size(l, h, v, s):
    P = 0
    # embedding
    P += (v + s) * h
    # attention
    P += (4 * h * h + 4 * h) * l
    # layer_norm of decoder
    P += (2 * (2 * h)) * l
    # FFN Layer
    P += (8 * h * h + 5 * h) * l
    # layer_norm of transformer
    P += 2 * h
    logger.info('Model Size: {:.2f} B'.format(P / 1000.0 / 1000.0 / 1000.0))


def build_model(config):
    nranks = dist.get_world_size()
    model_setting = copy.deepcopy(config.Model)

    if 'Compress' in config and 'Quantization' in config.Compress:
        quant_setting = copy.deepcopy(config.Compress.Quantization)
        model_setting['skip_tensor_map'] = quant_setting.get('skip_tensor_map',
                                                             {})
        model_setting['freeze_embedding'] = quant_setting.get(
            'freeze_embedding', False)

    l = model_setting['num_layers']
    h = model_setting['hidden_size']
    v = model_setting['vocab_size']
    s = config.Data.Train.dataset.max_seq_len
    _get_model_size(l, h, v, s)

    model_name = model_setting.pop("name")
    tokenizer_class, pretrained_name = MODEL_CLASSES[model_name]
    tokenizer = tokenizer_class.from_pretrained(pretrained_name)

    if nranks == 1:
        model_setting.pop("sequence_parallel")
        model = gpt.GPTForPretraining(gpt.GPTModel(**model_setting))
    else:
        model_setting['num_partitions'] = config.Distributed.mp_degree
        if config.Distributed.pp_degree == 1:
            model_setting.pop("virtual_pp_degree", None)
            model = gpt.GPTForPretrainingHybrid(
                gpt.GPTModelHybrid(**model_setting))
        else:
            model = gpt.GPTForPretrainingPipe(**model_setting)

    if config.Model.sequence_parallel:
        register_sequence_parallel_allreduce_hooks(
            model, config.Global.accumulate_steps,
            config.Distributed.fuse_sequence_parallel_allreduce)

    if nranks == 1:
        loss_fn = gpt.GPTPretrainingCriterion()
    else:
        loss_fn = gpt.GPTPretrainingCriterionHybird(
            sequence_parallel=config.Model.sequence_parallel)

    return model, tokenizer, loss_fn


def model_forward_backward(config, batch, forward_func, **kwargs):
    acc_steps = config.Global.accumulate_steps
    use_fp16 = config.Global.mix_precision.enable
    black_list = config.Global.mix_precision.custom_black_list
    white_list = config.Global.mix_precision.custom_white_list

    # train with pipeline strategy
    if config.Distributed.pp_degree > 1:
        tokens, position_ids, labels, loss_mask = batch
        batch = [(tokens, position_ids), (labels, loss_mask)]

        batches = [batch]

        with paddle.amp.auto_cast(
                use_fp16,
                custom_black_list=black_list,
                custom_white_list=white_list,
                level='O2'):

            batch = kwargs['model']._prepare_training(
                batch, kwargs['optimizer'], None)
            loss = kwargs['model'].forward_backward_pipeline(batch,
                                                             kwargs['scaler'])

        return loss

    # train with non-pipeline strategy
    if acc_steps == 1:
        batches = [batch]
    else:
        split_batches = [paddle.split(b, acc_steps) for b in batch]
        batches = []
        for i in range(len(split_batches[0])):
            micro_batch = [split_batch[i] for split_batch in split_batches]
            batches.append(micro_batch)

    # gradient merge strategy
    final_loss = None
    for micro_batch in batches:
        with paddle.amp.auto_cast(
                use_fp16,
                custom_black_list=black_list,
                custom_white_list=white_list,
                level='O2'):

            # forward in training step
            loss = forward_func(micro_batch, kwargs['model'],
                                kwargs['loss_fn'])

        loss_bw = kwargs['scaler'].scale(loss) if use_fp16 else loss
        loss_bw = loss_bw / acc_steps if acc_steps > 1 else loss_bw
        loss_bw.backward()

        detach_loss = loss.detach()
        if final_loss is None:
            final_loss = detach_loss
        else:
            final_loss = paddle.add(final_loss, detach_loss)

    final_loss = final_loss / acc_steps if acc_steps > 1 else final_loss

    return final_loss


def optim_update_params(config, **kwargs):
    hcg = env.get_hcg()
    use_fp16 = config.Global.mix_precision.enable

    dp_degree = config.Distributed.dp_degree
    sharding_stage = config.Distributed.sharding.sharding_stage

    if config.Model.use_recompute and isinstance(kwargs['model'],
                                                 paddle.DataParallel):
        if not hasattr(kwargs['optimizer'], "all_fused_tensors") or kwargs[
                'optimizer'].all_fused_tensors is None:
            fused_allreduce_gradients(list(kwargs['model'].parameters()), None)
        else:
            dp_group = hcg.get_data_parallel_group()
            all_reduce_parameters(kwargs['optimizer'].all_fused_tensors,
                                  dp_group)

    if sharding_stage == 3 and dp_degree > 1:
        dp_group = hcg.get_data_parallel_group()
        fused_allreduce_gradients(kwargs['model'].parameters(), hcg)

        for p in kwargs['model'].parameters():
            if hasattr(p, "bw_storage"):
                assert p.grad is None, "This case shouldn't happen."
                p.bw_storage.scale_(1.0 / dp_group.nranks)
                dist.all_reduce(p.bw_storage, group=dp_group)

    if use_fp16:
        kwargs['scaler'].step(kwargs['optimizer'])
        kwargs['scaler'].update()
    else:
        kwargs['optimizer'].step()


def fit_impl(config, batch, forward_func, **kwargs):
    kwargs['model'].train()

    if config.Distributed.pp_degree == 1:
        if config.Model.use_recompute and isinstance(kwargs['model'],
                                                     paddle.DataParallel):
            with kwargs['model'].no_sync():
                loss = model_forward_backward(config, batch, forward_func,
                                              **kwargs)
        else:
            loss = model_forward_backward(config, batch, forward_func,
                                          **kwargs)
    else:
        loss = model_forward_backward(config, batch, forward_func, **kwargs)

    optim_update_params(config, **kwargs)

    return loss


@paddle.no_grad()
def eval_impl(config, batch, model, loss_fn):
    model.eval()

    use_fp16 = config.Global.mix_precision.enable
    black_list = config.Global.mix_precision.custom_black_list
    white_list = config.Global.mix_precision.custom_white_list

    with paddle.amp.auto_cast(
            use_fp16,
            custom_black_list=black_list,
            custom_white_list=white_list,
            level='O2'):
        tokens, position_ids, labels, loss_mask = batch

        if config.Distributed.pp_degree == 1:
            tokens, position_ids, labels, loss_mask = batch
            preds = model(tokens, position_ids)
            preds = paddle.cast(preds, dtype="float32")
            loss = loss_fn(preds, labels, loss_mask)
        else:
            batch = [(tokens, position_ids), (labels, loss_mask)]
            loss = model.eval_batch(batch, compute_loss=True)

    return loss
