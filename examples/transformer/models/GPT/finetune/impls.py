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
import numpy as np

import paddle
import paddle.distributed as dist

from ppfleetx.utils.log import logger
from ppfleetx.distributed.apis import env
import ppfleetx.models.language_model.gpt as gpt
from ppfleetx.data.tokenizers import GPTTokenizer, GPTChineseTokenizer
from examples.transformer.models.GPT.pretrain.impls import fit_impl as pretrain_fit_impl

MODEL_CLASSES = {
    "GPT": (GPTTokenizer, "gpt2"),
    "GPT-cn": (GPTChineseTokenizer, "gpt-cpm-large-cn"),
}


def _get_model_size(l, h, v, s):
    P = 12 * l * h * h * (1 + 13 / (12 * h) + (v + s) / (12 * l * h))
    logger.info('Model Size: {:.2f} B'.format(P / 1000.0 / 1000.0 / 1000.0))


def build_model(config):
    nranks = dist.get_world_size()
    model_setting = copy.deepcopy(config.Model)

    loss_config = model_setting.pop("loss", None)
    metric_config = model_setting.pop("metric", None)
    pretrained = model_setting.pop("pretrained")
    num_classes = model_setting.pop("num_classes", 2)
    assert pretrained is not None

    l = model_setting['num_layers']
    h = model_setting['hidden_size']
    v = model_setting['vocab_size']
    num_heads = model_setting['num_attention_heads']
    s = config.Data.Train.dataset.max_length
    _get_model_size(l, h, v, s)

    model_name = model_setting.pop("name")
    tokenizer_class, pretrained_name = MODEL_CLASSES[model_name]
    tokenizer = tokenizer_class.from_pretrained(pretrained_name)

    if nranks == 1:
        model = gpt.GPTForSequenceClassification(
            gpt.GPTModel(**model_setting), num_classes)
    else:
        raise NotImplementedError

    pretrained_path = pretrained + ".pdparams"
    assert os.path.exists(pretrained_path), f'{pretrained_path} is not exists!'
    model_dict = paddle.load(pretrained_path)

    # Note(GuoxiaWang): Guess whether to convert fused vs non-fused parameters.
    # 'q_proj' vs 'qkv_proj'
    def is_fused(model_state):
        for key in model_state:
            if 'qkv_proj' in key:
                return True
        return False

    def split_params(model_state, num_layers):
        for idx in range(num_layers):
            qkv_b = model_state.pop(
                f'gpt.decoder.layers.{idx}.self_attn.qkv_proj.bias')
            qkv_w = model_state.pop(
                f'gpt.decoder.layers.{idx}.self_attn.qkv_proj.weight')

            qkv_b = qkv_b.reshape((num_heads, 3, -1))
            qkv_w = qkv_w.reshape((h, num_heads, 3, -1))

            q_w, k_w, v_w = np.split(qkv_w, 3, axis=2)
            q_w = q_w.reshape((h, -1))
            k_w = k_w.reshape((h, -1))
            v_w = v_w.reshape((h, -1))

            q_b, k_b, v_b = np.split(qkv_b, 3, axis=1)
            q_b = q_b.reshape((-1))
            k_b = k_b.reshape((-1))
            v_b = v_b.reshape((-1))

            model_state[
                f'gpt.decoder.layers.{idx}.self_attn.q_proj.bias'] = q_b
            model_state[
                f'gpt.decoder.layers.{idx}.self_attn.q_proj.weight'] = q_w

            model_state[
                f'gpt.decoder.layers.{idx}.self_attn.k_proj.bias'] = k_b
            model_state[
                f'gpt.decoder.layers.{idx}.self_attn.k_proj.weight'] = k_w

            model_state[
                f'gpt.decoder.layers.{idx}.self_attn.v_proj.bias'] = v_b
            model_state[
                f'gpt.decoder.layers.{idx}.self_attn.v_proj.weight'] = v_w

        return model_state

    def fuse_params(model_state, num_layers):
        for idx in range(num_layers):
            q_b = model_state.pop(
                f'gpt.decoder.layers.{idx}.self_attn.q_proj.bias')
            q_w = model_state.pop(
                f'gpt.decoder.layers.{idx}.self_attn.q_proj.weight')

            k_b = model_state.pop(
                f'gpt.decoder.layers.{idx}.self_attn.k_proj.bias')
            k_w = model_state.pop(
                f'gpt.decoder.layers.{idx}.self_attn.k_proj.weight')

            v_b = model_state.pop(
                f'gpt.decoder.layers.{idx}.self_attn.v_proj.bias')
            v_w = model_state.pop(
                f'gpt.decoder.layers.{idx}.self_attn.v_proj.weight')

            q_w = q_w.reshape((h, num_heads, -1))
            k_w = k_w.reshape((h, num_heads, -1))
            v_w = v_w.reshape((h, num_heads, -1))

            qkv_w = np.stack([q_w, k_w, v_w], axis=2)
            qkv_w = qkv_w.reshape((h, -1))

            q_b = q_b.reshape((num_heads, -1))
            k_b = k_b.reshape((num_heads, -1))
            v_b = v_b.reshape((num_heads, -1))
            qkv_b = np.stack([q_b, k_b, v_b], axis=1)
            qkv_b = qkv_b.reshape((-1))

            model_state[
                f'gpt.decoder.layers.{idx}.self_attn.qkv_proj.weight'] = qkv_w
            model_state[
                f'gpt.decoder.layers.{idx}.self_attn.qkv_proj.bias'] = qkv_b
        return model_state

    fused = is_fused(model.state_dict())
    load_fused = is_fused(model_dict)

    if fused is True and load_fused is False:
        model_dict = fuse_params(model_dict, l)
    elif fused is False and load_fused is True:
        model_dict = split_params(model_dict, l)

    for name, param in model.state_dict().items():
        if name in model_dict and param.dtype != model_dict[name].dtype:
            model_dict[name] = model_dict[name].cast(param.dtype)

    model.set_state_dict(model_dict)
    logger.info(f'Load pretrained weight from {pretrained_path}')

    # build loss fn
    assert loss_config is not None
    assert 'train' in loss_config and 'eval' in loss_config

    train_loss = copy.deepcopy(loss_config.train)
    train_loss_cls = train_loss.pop('name')
    train_loss_fn = eval(f'paddle.nn.loss.{train_loss_cls}')(**train_loss)

    eval_loss = copy.deepcopy(loss_config.eval)
    eval_loss_cls = eval_loss.pop('name')
    eval_loss_fn = eval(f'paddle.nn.loss.{eval_loss_cls}')(**eval_loss)

    return model, tokenizer, train_loss_fn, eval_loss_fn


def fit_impl(config, batch, forward_func, **kwargs):
    kwargs['model'].train()
    loss = pretrain_fit_impl(config, batch, forward_func, **kwargs)

    return loss


@paddle.no_grad()
def eval_impl(config, batch, model, loss_fn, eval_metric):
    model.eval()

    use_fp16 = config.Global.mix_precision.enable
    black_list = config.Global.mix_precision.custom_black_list
    white_list = config.Global.mix_precision.custom_white_list

    with paddle.amp.auto_cast(
            use_fp16,
            custom_black_list=black_list,
            custom_white_list=white_list,
            level='O2'):
        input_ids, labels = batch

        input_ids.stop_gradient = True
        labels.stop_gradient = True

        logits = model(input_ids)
        loss = loss_fn(logits, labels)
        correct = eval_metric.compute(logits, labels)
        eval_metric.update(correct)

    return loss
