# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid

input_x = fluid.layers.data()
input_y = fluid.layers.data()

fc_1 = fluid.layers.fc(input=input_x)
fc_2 = fluid.layers.fc(input=fc_1)
prediction = fluid.layers.fc(input=[fc_2])
cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
optimizer = fluid.optimizer.Adagrad(learning_rate=0.01)

role = UserDefinedRoleMaker()
fleet.init(role)

optimizer = fleet.distribute_optimize(optimizer)
optimizer.minimize(cost)

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
elif fleet.is_worker():
    pass_num = 10
    


