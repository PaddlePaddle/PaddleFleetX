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

import paddle
from paddle.distributed import fleet
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppfleetx.utils import config, env
from ppfleetx.utils.log import logger
from ppfleetx.data import build_dataloader
from ppfleetx.models import build_module
from ppfleetx.core import EagerEngine

#init_logger()

if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)

    if dist.get_world_size() > 1:
        fleet.init(is_collective=True, strategy=env.init_dist_env(cfg))

    env.set_seed(cfg.Global.seed)

    module = build_module(cfg)
    config.print_config(cfg)

    train_data_loader = build_dataloader(cfg.Data, "Train")
    eval_data_loader = build_dataloader(cfg.Data, "Eval")

    cfg.Optimizer.lr.update({
        'epochs': cfg.Engine.num_train_epochs,
        'step_each_epoch': len(train_data_loader),
        'total_steps': cfg.Engine.max_steps,
    })

    engine = EagerEngine(configs=cfg, module=module)

    if cfg.Engine.save_load.ckpt_dir is not None:
        engine.load()

    def _evaluate_one_epoch():
        eval_losses = []
        total_eval_batch = len(eval_data_loader)
        for eval_step, batch in enumerate(eval_data_loader):
            loss = _evaluate_impl(batch)

            paddle.device.cuda.synchronize()
            eval_losses.append(loss.numpy()[0])

        return sum(eval_losses) / len(eval_losses)

    def _evaluate_impl(batch):
        batch = engine._module.pretreating_batch(batch)

        with paddle.amp.auto_cast(
                engine._use_pure_fp16,
                custom_black_list=engine._custom_black_list,
                custom_white_list=engine._custom_white_list,
                level='O2'):
            if engine._pp_degree == 1:
                loss = engine._module.validation_step(batch)
            else:
                loss = engine._module.model.eval_batch(batch, compute_loss=True)
        return loss


    if "Prune" in cfg.keys() and cfg.Prune.enable:
        engine.prune_model()

    if 'Quantization' in cfg.keys() and cfg.Quantization.enable:
        engine.quant_model()

    engine.distributed_model()
    if cfg.Engine.save_load.save_only:
        engine.save(0, 0)
        sys.exit()

    if "Prune" in cfg.keys() and cfg.Prune.cal_sens:
        engine.sensitive(_evaluate_one_epoch)
        sys.exit()

    engine.fit(train_data_loader=train_data_loader,
               valid_data_loader=eval_data_loader,
               epoch=cfg.Engine.num_train_epochs)
