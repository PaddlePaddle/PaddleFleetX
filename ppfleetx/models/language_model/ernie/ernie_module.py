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
sys.path.append("../../../../../")
from ppfleetx.core.module.basic_module import BasicModule
import ppfleetx.models.language_model.gpt as gpt
from ppfleetx.utils import logger
from .single_model import ErnieModel, ErnieForPretraining
from ppfleetx.models.language_model.utils import process_configs


class ErnieModule(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()
        super(ErnieModule, self).__init__(configs)

    # def process_configs(self, configs):
    #     configs = process_configs(configs)
    #     return configs

    def get_loss_fn(self):
        return None

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        print(model_setting)
        model_setting.pop("module")
        model_setting.pop("name")
        model = ErnieForPretraining(ErnieModel(**model_setting))
        return model
