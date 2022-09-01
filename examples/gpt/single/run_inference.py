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

sys.path.append("../../../")
import paddle
from paddle.distributed import fleet

from examples.gpt.gpt_module import GPTGenerationModule
from fleetx.data.tokenizers import GPTTokenizer
from fleetx.core.engine.eager_engine import EagerEngine

from examples.gpt.tools import parse_yaml


def do_inference():
    configs = parse_yaml()

    paddle.set_device(configs['Global']['device'])

    seed = configs['Global']['seed']
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    module = GPTGenerationModule(configs)
    engine = EagerEngine(module=module, configs=configs, mode='test')

    input_text = 'Hi, GPT2. Tell me who Jack Ma is.'
    input_ids = [tokenizer.encode(input_text)]

    outs = engine.inference([input_ids])

    ids = list(outs.values())[0]
    out_ids = [int(x) for x in ids[0]]
    result = tokenizer.decode(out_ids)
    result = input_text + result

    print('Prompt:', input_text)
    print('Generation:', result)


if __name__ == "__main__":
    do_inference()
