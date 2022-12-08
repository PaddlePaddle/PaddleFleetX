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

from ppfleetx.utils.log import logger
import ppfleetx.models.language_model.gpt as gpt
from ppfleetx.optims.optimizer import *
from ppfleetx.optims.grad_clip import *
from ppfleetx.optims.lr_scheduler import *
from ppfleetx.data.tokenizers import GPTTokenizer, GPTChineseTokenizer
from ppfleetx.models.language_model.gpt.dygraph.sequence_parallel_utils import register_sequence_parallel_allreduce_hooks

MODEL_CLASSES = {
    "GPT": (GPTTokenizer, "gpt2"),
    "GPT-cn": (GPTChineseTokenizer, "gpt-cpm-large-cn"),
}


def get_model_size(l, h, v, s):
    P = 12 * l * h * h * (1 + 13 / (12 * h) + (v + s) / (12 * l * h))
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
    get_model_size(l, h, v, s)

    model_name = model_setting.pop("name")
    tokenizer_class, pretrained_name = MODEL_CLASSES[model_name]
    tokenizer = tokenizer_class.from_pretrained(pretrained_name)

    moe_configs = model_setting.get('moe_configs', {'expert_mode': False})
    assert not moe_configs[
        'expert_mode'], "Not support expert mode in GPT model!"
    model_setting["moe_configs"] = moe_configs

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
            model, config.Engine.accumulate_steps,
            config.Distributed.fuse_sequence_parallel_allreduce)

    if nranks == 1:
        loss_fn = gpt.GPTPretrainingCriterion()
    else:
        loss_fn = gpt.GPTPretrainingCriterionHybird(
            sequence_parallel=config.Model.sequence_parallel)

    return model, tokenizer, loss_fn


def build_lr_scheduler(lr_config):
    if 'name' in lr_config:
        lr_name = lr_config.pop('name')
        lr = eval(lr_name)(**lr_config)
        if isinstance(lr, LRScheduler):
            return lr
        else:
            return lr()
    else:
        lr = lr_config.learning_rate

    logger.debug("build lr ({}) success..".format(lr))
    return lr


def build_grad_clip(grad_clip_config):
    if grad_clip_config is not None:
        grad_clip_name = grad_clip_config.pop('name', 'ClipGradByGlobalNorm')
        grad_clip = eval(grad_clip_name)(**grad_clip_config)
        return grad_clip
    else:
        return None


def build_optimizer(config, model, lr_scheduler=None, multi_precision=False):
    config = copy.deepcopy(config)
    if lr_scheduler is not None:
        config.pop('lr')

    grad_clip_config = config.pop('grad_clip', None)
    grad_clip = build_grad_clip(grad_clip_config)

    optim_name = config.pop('name')
    optim = eval(optim_name)(learning_rate=lr_scheduler,
                             parameters=model.parameters(),
                             grad_clip=grad_clip,
                             multi_precision=multi_precision,
                             **config)

    logger.debug("build optimizer ({}) success..".format(optim))
    return optim
