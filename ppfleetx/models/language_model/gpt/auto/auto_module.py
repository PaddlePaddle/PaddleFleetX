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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import argparse

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import LazyGuard
#from paddle.distributed.fleet import auto

from .auto_utils import process_configs

import ppfleetx.models.language_model.gpt as gpt
from ppfleetx.utils.log import logger
from ppfleetx.data.tokenizers import GPTTokenizer
from ppfleetx.core.module.basic_module import BasicModule


class LanguageModuleAuto(BasicModule):
    def __init__(self, configs):
        self.nranks = dist.get_world_size()
        super(LanguageModuleAuto, self).__init__(configs)

        self.loss_fn = self.get_loss_fn()

    def process_configs(self, configs):
        configs = process_configs(configs)
        return configs

    def get_model_size(self, l, h, v, s):
        P = 12 * l * h * h * (1 + 13 / (12 * h) + (v + s) / (12 * l * h))
        logger.info('Model Size: {:.2f} B'.format(P / 1000.0 / 1000.0 /
                                                  1000.0))


class GPTModuleAuto(LanguageModuleAuto):
    def __init__(self, configs):
        super(GPTModuleAuto, self).__init__(configs)

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        l = model_setting['num_layers']
        h = model_setting['hidden_size']
        v = model_setting['vocab_size']
        s = self.configs.Data.Train.dataset.max_seq_len
        self.get_model_size(l, h, v, s)

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        with LazyGuard():
            model = gpt.GPTForPretrainingAuto(
                gpt.GPTModelAuto(**model_setting))
        return model

    def get_loss_fn(self):
        model_setting = copy.deepcopy(self.configs.Model)
        return gpt.GPTPretrainingCriterionAuto(model_setting['mesh'])
