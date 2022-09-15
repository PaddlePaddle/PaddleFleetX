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
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

sys.path.append("../../../")
from fleetx.core.engine.eager_engine import EagerEngine
from fleetx.data.tokenizers import GPTTokenizer
from examples.gpt.gpt_module import GPTGenerationModule
from examples.gpt.tools import parse_yaml


def set_hyrbid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank):
    random.seed(basic_seed + data_world_rank)
    np.random.seed(basic_seed + data_world_rank)
    paddle.seed(basic_seed + data_world_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = basic_seed + 123 + mp_rank * 10 + pp_rank * 1000
    global_seed = basic_seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add('global_seed', global_seed)
    tracker.add('local_seed', local_seed)


def do_export():
    configs = parse_yaml()

    paddle.set_device(configs['Global']['device'])

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": configs['Distributed']['dp_degree'],
        "mp_degree": configs['Distributed']['mp_degree'],
        "pp_degree": configs['Distributed']['pp_degree'],
        "sharding_degree":
        configs['Distributed']['sharding']['sharding_degree'],
    }

    strategy.pipeline_configs = {
        "accumulate_steps": configs['Engine']['accumulate_steps'],
        "micro_batch_size": configs['Data']['batch_size']['micro_batch_size']
    }

    seed = configs['Global']['seed']

    # set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
    fleet.init(is_collective=True, strategy=strategy)

    # obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    pp_rank = hcg.get_stage_id()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()
    sharding_size = hcg.get_sharding_parallel_world_size()

    data_world_rank = dp_rank * sharding_size + sharding_rank
    data_world_size = configs['Distributed']['dp_degree'] * \
        configs['Distributed']['sharding']['sharding_degree']
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(seed, data_world_rank, mp_rank, pp_rank)

    module = GPTGenerationModule(configs)
    engine = EagerEngine(module=module, configs=configs, mode='export')

    tokenizer = GPTTokenizer.from_pretrained("gpt2")
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
    do_export()
