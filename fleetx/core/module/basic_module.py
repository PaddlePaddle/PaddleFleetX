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
import paddle.nn as nn


class BasicModule(nn.Layer):
    """

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    def training_step_end(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def validation_step_end(self, *args, **kwargs):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def test_step_end(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        raise NotImplementedError

    def backward(self, loss):
        loss.backward()
