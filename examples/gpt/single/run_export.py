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
from examples.gpt.gpt_module import GPTGenerationModule, GPTModule
from examples.gpt.tools import parse_yaml
from fleetx.datasets.gpt import create_pretrained_dataset, get_train_data_file
from fleetx.data.tokenizers import GPTTokenizer
from fleetx.core.engine.eager_engine import EagerEngine


def do_export():
    configs = parse_yaml()

    paddle.set_device(configs['Global']['device'])

    seed = configs['Global']['seed']
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    tokenizer = GPTTokenizer.from_pretrained("gpt2")

    module = GPTGenerationModule(configs)
    # module = GPTModule(configs)

    engine = EagerEngine(module=module, configs=configs, mode='export')

    ckpt_dir = configs['Engine']['save_load']['ckpt_dir']
    if ckpt_dir is None or not os.path.isdir(ckpt_dir):
        raise ValueError("config ckpt_dir invalid: {}".format(ckpt_dir))

    # FIXME(dengkaipeng): change to engine.load after engine.load fixed
    model_dict = paddle.load(os.path.join(ckpt_dir, 'model.pdparams'))
    engine._module.model.set_state_dict(model_dict)

    engine.export()


if __name__ == "__main__":
    do_export()
