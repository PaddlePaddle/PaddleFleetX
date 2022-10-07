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

paddle.enable_static()

exe = paddle.static.Executor()
[inference_program, feed_target_names,
 fetch_targets] = paddle.static.load_inference_model(
     path_prefix="./output/auto_dist0", executor=exe)

print(feed_target_names)
print(fetch_targets)
print(inference_program)
