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

import sys
import copy

sys.path.append("../../")
from ppfleetx.core.module.basic_module import BasicModule
from ppfleetx.models.language_model.language_module import GPTModule, MoEModule
from ppfleetx.models.language_model.ernie import ErnieModule


def build_module(config):
    module_name = config.Model.get("module", "BasicModule")
    module = eval(module_name)(config)

    return module
