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

import os
import random
import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

__all__ = ['init_dist_env']


def set_seed(seed):
    if dist.get_world_size() > 1:
        # obtain rank message of hybrid parallel
        hcg = fleet.get_hybrid_communicate_group()
        mp_rank = hcg.get_model_parallel_rank()
        pp_rank = hcg.get_stage_id()
        data_world_rank = get_data_world_rank()
    else:
        mp_rank, pp_rank, data_world_rank = 1, 1, 1

    random.seed(seed + data_world_rank)
    np.random.seed(seed + data_world_rank)
    paddle.seed(seed + data_world_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = seed + 123 + mp_rank * 10 + pp_rank * 1000
    global_seed = seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add('global_seed', global_seed)
    tracker.add('local_seed', local_seed)


def init_dist_env(config):
    paddle.set_device(config.Global.device)

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": config.Distributed.dp_degree,
        "mp_degree": config.Distributed.mp_degree,
        "pp_degree": config.Distributed.pp_degree,
        "sharding_degree": config.Distributed.sharding.sharding_degree,
    }

    strategy.pipeline_configs = {
        "accumulate_steps": config.Engine.accumulate_steps,
        "micro_batch_size": config.Global.micro_batch_size,
    }

    # set control in tensor parallel
    seed = config.Global.seed
    strategy.tensor_parallel_configs = {"tensor_init_seed": seed}

    return strategy


def get_local_rank():
    return int(os.getenv("PADDLE_RANK_IN_NODE", 0))


def get_data_world_size():
    if paddle.distributed.get_world_size() == 1:
        return 1

    hcg = fleet.get_hybrid_communicate_group()
    dp_size = hcg.get_data_parallel_world_size()
    sharding_size = hcg.get_sharding_parallel_world_size()

    return dp_size * sharding_size


def get_data_world_rank():
    if paddle.distributed.get_world_size() == 1:
        return 0

    hcg = fleet.get_hybrid_communicate_group()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()
    sharding_size = hcg.get_sharding_parallel_world_size()

    return dp_rank * sharding_size + sharding_rank
