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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from paddle.distributed import fleet
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppfleetx.utils import config, env
from ppfleetx.utils.logger import init_logger
from ppfleetx.data import build_dataloader
from ppfleetx.models import build_module
from ppfleetx.optims import build_lr_scheduler, build_optimizer
from ppfleetx.core import EagerEngine

init_logger()

if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)

    if dist.get_world_size() > 1:
        fleet.init(is_collective=True, strategy=env.init_dist_env(cfg))

    env.set_seed(cfg.Global.seed)

    module = build_module(cfg)
    config.print_config(cfg)

    lr = build_lr_scheduler(cfg.Optimizer.lr)
    optimizer = build_optimizer(cfg.Optimizer, module.model, lr)

    engine = EagerEngine(
        configs=cfg, module=module, optimizer=optimizer, lr=lr)

    if cfg.Engine.save_load.ckpt_dir is not None:
        engine.load()

    train_data_loader = build_dataloader(cfg.Data, "Train")
    eval_data_loader = build_dataloader(cfg.Data, "Eval")

    engine.fit(train_data_loader=train_data_loader,
               valid_data_loader=eval_data_loader,
               epoch=cfg.Engine.num_train_epochs)
