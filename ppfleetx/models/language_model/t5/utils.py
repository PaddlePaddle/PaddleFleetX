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

import paddle
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

trunc_normal_ = TruncatedNormal(std=0.02)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


@paddle.no_grad()
def constant_(x, value):
    temp_value = paddle.full(x.shape, value, x.dtype)
    x.set_value(temp_value)
    return x


@paddle.no_grad()
def normal_(x, mean=0., std=1.):
    temp_value = paddle.normal(mean, std, shape=x.shape)
    x.set_value(temp_value)
    return


def normal_init(layer, mean=0, std=1, bias=0):
    if hasattr(layer, 'weight') and layer.weight is not None:
        normal_(layer.weight, mean, std)
    else:
        normal_(layer, mean, std)
    if hasattr(layer, 'bias') and layer.bias is not None:
        constant_(layer.bias, bias)


def constant_init(layer, val, bias=0):
    if hasattr(layer, 'weight') and layer.weight is not None:
        constant_(layer.weight, val)
    if hasattr(layer, 'bias') and layer.bias is not None:
        constant_(layer.bias, bias)
