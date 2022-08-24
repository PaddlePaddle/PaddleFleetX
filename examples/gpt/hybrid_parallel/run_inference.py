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

from examples.gpt.gpt_module import GPTModule
from fleetx.data.tokenizers import GPTTokenizer
from fleetx.core.engine.eager_engine import EagerEngine
from fleetx.datasets.gpt import create_pretrained_dataset, get_train_data_file

from examples.gpt.tools import parse_args, parse_yaml


def do_inference():
    configs = parse_yaml(parse_args())

    paddle.set_device(configs['Global']['device'])

    seed = configs['Global']['seed']
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    module = GPTModule(configs)
    engine = EagerEngine(module=module, configs=configs)

    files = get_train_data_file(configs['Data']['dataset']['input_dir'])
    data_file = np.random.choice(files)
    _, _, test_data_loader = create_pretrained_dataset(
        configs, [data_file],
        local_rank=0,
        data_world_size=1,
        data_world_rank=0,
        max_seq_len=configs['Data']['dataset']['max_seq_len'],
        eos_id=tokenizer.eos_token_id)

    for iter_id, data in enumerate(test_data_loader):
        outs = engine.inference(data)
        for k, v in outs.items():
            for i in range(v.shape[0]):
                out_ids = [int(x) for x in v[i]]
                ret_str = tokenizer.decode(out_ids)
                ret_str = text[i] + ret_str
                print(ret_str)


if __name__ == "__main__":
    do_inference()
