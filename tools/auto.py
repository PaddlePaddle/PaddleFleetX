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
import copy
import random
import paddle
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppfleetx.utils import config
from ppfleetx.utils.log import logger
from ppfleetx.models import build_module
from ppfleetx.data import build_dataset
from ppfleetx.optims import build_lr_scheduler, build_optimizer
from ppfleetx.core import AutoEngine

if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_auto_config(
        args.config, overrides=args.override, show=False)

    module = build_module(cfg)
    config.print_config(cfg)

    train_data = build_dataset(cfg.Data, "Train")
    eval_data = build_dataset(cfg.Data, "Eval")

    lr_configs = copy.deepcopy(cfg.Optimizer.lr)
    lr_configs.update({
        'epochs': cfg.Engine.num_train_epochs,
        'step_each_epoch': len(train_data)
    })
    lr = build_lr_scheduler(lr_configs)
    optimizer = build_optimizer(cfg.Optimizer, module.model, lr)

    engine = AutoEngine(configs=cfg, module=module, optimizer=optimizer, lr=lr)

    if cfg.Engine.save_load.ckpt_dir is not None:
        engine.load()

    engine.fit(train_dataset=train_data,
               valid_dataset=eval_data,
               epoch=cfg.Engine.num_train_epochs)