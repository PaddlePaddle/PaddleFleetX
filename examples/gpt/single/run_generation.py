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

import argparse
import math
import os
import random
import time
import sys
import yaml
import numpy as np

import paddle
sys.path.append("../../../")
from examples.gpt.gpt_module import GPTGenerationModule
from examples.gpt.tools import parse_args, parse_yaml
from fleetx.core.engine.eager_engine import EagerEngine


def do_generation():
    configs = parse_yaml(parse_args())

    paddle.set_device(configs['Global']['device'])

    seed = configs['Global']['seed']

    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    module = GPTGenerationModule(configs)

    engine = EagerEngine(module=module, configs=configs, mode='predict')
    engine.load()

    input_text = 'Where are you from?'

    result = engine.predict(input_text)

    print(result[0])


if __name__ == "__main__":
    do_generation()
