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
import sys
import copy

import paddle
from paddle.distributed import fleet
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../../../')))

from ppfleetx.utils.log import logger
from ppfleetx.data import build_dataloader
from ppfleetx.core import EagerEngine
from ppfleetx.distributed.apis import env, fleet_wrapper
from examples.transformer.utils import config as cfg

import modeling

if __name__ == "__main__":
    args = cfg.parse_args()
    config = cfg.get_config(args.config, overrides=args.override, show=False)

    paddle.set_device(config.Global.device)

    nranks = dist.get_world_size()
    if nranks > 1:
        env.init_dist_env(config)

    env.set_seed(config.Global.seed)

    cfg.process_configs(config)
    cfg.print_config(config)

    model, tokenizer, loss_fn = modeling.build_model(config)

    if config.Global.mix_precision.use_pure_fp16:
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=config.Global.mix_precision.scale_loss)
        # Save dtype is the same as model dtype. Also can set save_dtype='float32' when 
        # training with pure fp16 strategy, but will cause the rise of memory.
        model = paddle.amp.decorate(models=model, level='O2')
    else:
        scaler = None

    # config.Optimizer.lr.update({
    #     'epochs': config.Global.num_train_epochs,
    #     'step_each_epoch': len(train_data_loader),
    #     'total_steps': config.Global.max_steps,
    # })

    lr_scheduler = modeling.build_lr_scheduler(config.Optimizer.lr)
    optimizer = modeling.build_optimizer(
        config.Optimizer,
        model,
        lr_scheduler,
        multi_precision=config.Global.mix_precision.use_pure_fp16)

    # distributed configs
    if nranks > 1:
        hcg = env.get_hcg()
        dp_group = hcg.get_data_parallel_group()
        sharding_group = hcg.get_sharding_parallel_group()

        dp_rank = hcg.get_data_parallel_rank()
        mp_rank = hcg.get_model_parallel_rank()
        pp_rank = hcg.get_stage_id()
        sharding_rank = hcg.get_sharding_parallel_rank()

        model, optimizer, scaler = fleet_wrapper.wrap_with_fleet(
            config.Distributed, model, optimizer, scaler)
    else:
        dp_rank = 0
