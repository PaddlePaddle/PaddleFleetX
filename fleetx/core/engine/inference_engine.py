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
import time
import sys

import paddle
import paddle.nn as nn
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

sys.path.append("../../../")
from fleetx.utils import logger
from fleetx.core.engine.basic_engine import BasicEngine
from fleetx.core.module.basic_module import BasicModule


class InferenceEngine(BasicEngine):
    """
    """

    def __init__(self, module, configs=None):
        super().__init__()

        if not isinstance(module, BasicModule):
            raise TypeError(
                "'module' must be sub classes of `BasicModule`, but got: {model.__class__.__name__}."
            )

        self._module = module

        # engine configs
        self._configs = configs['Engine']
        self._ckpt_dir = self._configs['save_load']['ckpt_dir']

        # TODO(haohongxiang): Remove there extra configs after reconstruct of Fleet API
        self._dist_configs = configs['Distributed']
        self._dp_degree = self._dist_configs['dp_degree']
        self._mp_degree = self._dist_configs['mp_degree']
        self._pp_degree = self._dist_configs['pp_degree']
        self._sharding_stage = self._dist_configs['sharding']['sharding_stage']
        self._sharding_degree = self._dist_configs['sharding'][
            'sharding_degree']
        self._sharding_offload = self._dist_configs['sharding'][
            'sharding_offload']
        self._use_recompute = configs['Model']['use_recompute']

        self._distributed = dist.is_initialized()

    @paddle.no_grad()
    def predict(self, inputs):
        self._module.model.eval()
        ret = self._module(inputs)
        return ret

    def load(self):
        if self._ckpt_dir and isinstance(self._ckpt_dir, str):
            logger.info("Try to load checkpoint from %s " % self._ckpt_dir)

            load_dir = "{}/mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}".format(
                self._ckpt_dir, self._mp_rank, self._sharding_rank,
                self._pp_rank) if self._distributed else self._ckpt_dir
            model_path = os.path.join(load_dir, "model.pdparams")
            if os.path.exists(model_path):
                model_dict = paddle.load(model_path)
                self._module.model.set_state_dict(model_dict)
            else:
                logger.warning("No model checkpoint file found in %s." %
                               model_path)
        else:
            logger.warning("`load` requires a valid value of `ckpt_dir`.")
