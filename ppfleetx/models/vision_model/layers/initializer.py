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

import numpy as np
import math
from paddle.nn.initializer import Constant, Normal, XavierUniform, Uniform

mlp_bias_normal_ = Normal(std=1e-6)
pos_normal_ = Normal(std=0.02)
xavier_uniform_ = XavierUniform()
zeros_ = Constant(value=0.)
minus_tens_ = Constant(value=-10.)
ones_ = Constant(value=1.)


class XavierUniform2D(XavierUniform):
    def __call__(self, param, block=None):
        fan_in = int(np.prod(param.shape[:-1]))
        fan_out = param.shape[-1]
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        uniform = Uniform(low=-limit, high=limit)
        uniform(param)


xavier_uniform_2d_ = XavierUniform2D()
