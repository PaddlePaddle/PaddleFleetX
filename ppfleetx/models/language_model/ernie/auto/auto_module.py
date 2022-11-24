#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import copy

import paddle
from paddle import LazyGuard
from ppfleetx.core.module.basic_module import BasicModule
from ppfleetx.utils.log import logger

from .auto_model import ErnieModelAuto, ErnieForPretrainingAuto, ErniePretrainingCriterionAuto

from ppfleetx.models.language_model.auto_utils import process_configs, process_mesh_config

import numpy as np


def process_data_configs(config):
    """
    process data configs for hybrid parallel
    """
    cfg_global = config['Global']
    cfg_data = config['Data']

    mode_to_num_samples = {
        "Train":
        cfg_global['global_batch_size'] * config['Engine']['max_steps'],
        "Eval": cfg_global['global_batch_size'] *
        (config['Engine']['max_steps'] // config['Engine']['eval_freq'] + 1) *
        config['Engine']['eval_iters'],
        "Test":
        cfg_global['global_batch_size'] * config['Engine']['test_iters'],
    }

    for mode in ("Train", "Eval", "Test"):
        if mode in cfg_data.keys():
            cfg_data[mode]['dataset']['num_samples'] = mode_to_num_samples[
                mode]
            cfg_data[mode]['dataset']['mode'] = mode
            cfg_data[mode]['dataset']['seed'] = cfg_global['seed']
            cfg_data[mode]['dataset'].setdefault('binary_head',
                                                 cfg_global['binary_head'])


def process_model_configs(config):
    cfg_model = config['Model']
    mesh = process_mesh_config(config['Distributed'])
    cfg_model.update({'mesh': mesh})
    cfg_model.setdefault("intermediate_size", cfg_model['hidden_size'] * 4)


class ErnieModuleAuto(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()
        super(ErnieModuleAuto, self).__init__(configs)
        self.nranks = paddle.distributed.get_world_size()
        self.binary_head = self.configs['Global']['binary_head']

        self.loss_fn = ErniePretrainingCriterionAuto(self.binary_head)

    def process_configs(self, configs):
        process_data_configs(configs)
        process_model_configs(configs)
        return configs

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        print("model_setting:", model_setting)
        with LazyGuard():
            model = ErnieForPretrainingAuto(ErnieModelAuto(**model_setting))

        return model

    def input_spec(self):
        inputs_spec = [
            paddle.static.InputSpec(
                shape=[None, None], name="input_ids", dtype="int64"),
            paddle.static.InputSpec(
                shape=[None, None], name="token_type_ids", dtype="int64"),
            paddle.static.InputSpec(
                shape=[None, None], name="position_ids", dtype="int64"),
        ]

        return inputs_spec
