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
from examples.gpt.tools import parse_yaml


def do_generation():
    configs = parse_yaml()

    paddle.set_device(configs['Global']['device'])

    seed = configs['Global']['seed']

    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    module = GPTGenerationModule(configs)
    module.eval()

    ckpt_dir = configs['Engine']['save_load']['ckpt_dir']

    model_path = os.path.join(ckpt_dir, "model.pdparams")
    model_dict = paddle.load(model_path)

    for key, value in model_dict.items():
        model_dict[key] = model_dict[key].astype(paddle.float32)

    module.model.set_state_dict(model_dict)

    input_text = 'Hi, GPT2. Tell me who Jack Ma is.'
    result = module.generate(input_text)

    print(f'Prompt: {input_text}')
    print(f'Generation: {result[0]}')


if __name__ == "__main__":
    do_generation()
