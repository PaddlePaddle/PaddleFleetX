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

import random
import sys
import numpy as np

import paddle
from paddle.distributed import fleet

sys.path.append("../../../")
from examples.gpt.tools import parse_args, parse_yaml_auto
from examples.gpt.gpt_module import GPTAutoModule
from fleetx.core.engine.static_engine import StaticEngine
from fleetx.data.tokenizers import GPTTokenizer
from fleetx.datasets.gpt import create_pretrained_dataset_auto, get_train_data_file


def generate_dist_strategy(configs):

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    dist_strategy.recompute = configs['Engine']['use_recompute']
    amp_configs = configs['Engine']['mix_precision']
    dist_strategy.amp = amp_configs['use_pure_fp16']
    dist_strategy.amp_configs = {
        "custom_black_list": amp_configs['custom_black_list'],
        "custom_white_list": amp_configs['custom_white_list'],
        "init_loss_scaling": amp_configs['scale_loss'],
        "use_pure_fp16": amp_configs['use_pure_fp16'],
    }
    return dist_strategy


def do_train():
    configs = parse_yaml_auto(parse_args().config)
    seed = configs['Global']['seed']
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    fleet.init(is_collective=True)
    dist_strategy = generate_dist_strategy(configs)
    tokenizer = GPTTokenizer.from_pretrained("gpt2")

    module = GPTAutoModule(configs)
    engine = StaticEngine(
        module=module, configs=configs, dist_strategy=dist_strategy)
    engine.load()

    for epoch in range(configs['Engine']['num_train_epochs']):
        files = get_train_data_file(configs['Data']['dataset']['input_dir'])
        files.sort()
        num_files = len(files)
        for f_id in range(num_files):
            data_file = files[f_id]
            train_data, _, _ = create_pretrained_dataset_auto(
                configs, [data_file], tokenizer.eos_token_id)
            engine.fit(train_dataset=train_data, epoch=epoch)


if __name__ == "__main__":
    do_train()
