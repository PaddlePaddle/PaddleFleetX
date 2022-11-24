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
import numpy as np

from paddle.distributed import fleet
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from ppfleetx.utils import config
from ppfleetx.distributed.apis import env
from ppfleetx.utils.log import logger
from ppfleetx.data import build_dataloader, tokenizers
from ppfleetx.models import build_module
from ppfleetx.core import EagerEngine

if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)
    env.set_seed(cfg.Global.seed)
    np.random.seed(1)
    input_ids = np.ones((16, 128), dtype="int64")
    position_ids = np.ones((16, 128), dtype="int64")
    type_ids = np.ones((16, 128), dtype="int64")

    if (os.path.exists('shape.pbtxt') == False):
        cfg.Inference.TensorRT.collect_shape = True
        module = build_module(cfg)
        engine = EagerEngine(configs=cfg, module=module, mode='inference')
        outs = engine.inference([input_ids, position_ids, type_ids])

    cfg.Inference.TensorRT.collect_shape = False
    module = build_module(cfg)
    config.print_config(cfg)
    engine = EagerEngine(configs=cfg, module=module, mode='inference')
    outs = engine.inference([input_ids, position_ids, type_ids])
    print(outs)
