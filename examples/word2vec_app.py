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

import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet

role = fleet.PaddleCloudRoleMaker()
fleet.init(role)

model = X.applications.Word2vec()

"""
need config loader correctly.
"""

loader = model.load_dataset_from_file(train_files_path=[], dict_path="")

dist_strategy = fleet.DistributedStrategy()
dist_strategy.a_sync = True

optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
else:
    trainer = X.CPUTrainer()
    trainer.fit(model, loader, epoch=10)
