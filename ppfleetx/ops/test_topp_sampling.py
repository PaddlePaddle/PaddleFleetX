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
import numpy as np
import custom_setup_ops

paddle.seed(2022)

x = paddle.randn([1, 51200], dtype="float16")
x = paddle.nn.functional.softmax(x)
top_ps = paddle.to_tensor(np.random.uniform(0, 1, [1]).astype(np.float16))
out = custom_setup_ops.topp_sampling(x, top_ps, 8)
print(out)
