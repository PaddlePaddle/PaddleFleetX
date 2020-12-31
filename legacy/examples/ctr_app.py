# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import fleetx as X
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker


configs = X.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)
model = X.applications.MultiSlotCTR()
loader = model.load_criteo_from_file('./ctr_data/train_data')

dist_strategy = fleet.DistributedStrategy()
dist_strategy.a_sync = True
dist_strategy.a_sync_configs = {"k_steps": 10000, "send_queue_size": 32}

optimizer = fluid.optimizer.SGD(learning_rate=0.0001)

optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
else:
    trainer = X.CPUTrainer()
    trainer.fit(model, loader, epoch=1)
