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


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 or length > max_sequence_length:
        length = max_sequence_length
    return length


def build_model(config):
    nranks = dist.get_world_size()
    generation_cfgs = config.Generation

    model_setting = copy.deepcopy(config.Model)
    if 'Compress' in config and 'Quantization' in config.Compress:
        quant_setting = copy.deepcopy(config.Compress.Quantization)
        skip_tensor_map = quant_setting.get('skip_tensor_map', {})
        freeze_embedding = quant_setting.get('freeze_embedding', False)
        model_setting['skip_tensor_map'] = skip_tensor_map
        model_setting['freeze_embedding'] = freeze_embedding

    model_name = model_setting.pop("name")
    tokenizer_class, pretrained_name = MODEL_CLASSES[model_name]
    tokenizer = tokenizer_class.from_pretrained(pretrained_name)

    if nranks == 1:
        model = gpt.GPTForGeneration(
            gpt.GPTModel(**model_setting), generation_cfgs)
    else:
        assert nranks == config.Distributed.dp_degree, \
            "only support single card and data parallel in generation task."
        model = gpt.GPTForGenerationHybrid(
            gpt.GPTModelHybrid(**model_setting), generation_cfgs)

    generation_cfgs['max_dec_len'] = adjust_length_to_model(
        generation_cfgs['max_dec_len'], 512)

    generation_cfgs['bos_token_id'] = tokenizer.eos_token_id
    generation_cfgs['eos_token_id'] = tokenizer.eos_token_id
    generation_cfgs['pad_token_id'] = tokenizer.eos_token_id

    return model, tokenizer


def left_padding(inputs, pad_id, padding="longest"):
    assert "input_ids" in inputs, "input_ids should be in inputs!"
    max_length = 0
    for ids in inputs["input_ids"]:
        max_length = max(max_length, len(ids))

    def extend_max_lenth(value, max_length, to_pad_id):
        return [to_pad_id] * (max_length - len(value)) + value

    def extend_filed(name, max_length, to_pad_id):
        values = inputs[name]
        res = []
        for index, value in enumerate(values):
            res.append(extend_max_lenth(value, max_length, to_pad_id))
        inputs[name] = res

    extend_filed("input_ids", max_length, pad_id)
    if "attention_mask" in inputs:
        extend_filed("attention_mask", max_length, 0)
    if "position_ids" in inputs:
        extend_filed("position_ids", max_length, 0)

    return inputs
