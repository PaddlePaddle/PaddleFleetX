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
import fleetx as X
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
paddle.enable_static()
configs = X.parse_train_configs()

fleet.init(is_collective=True)
model = X.applications.VGG16()
downloader = X.utils.Downloader()
local_path = downloader.download_from_hdfs(
    configs.download_config, local_path='.')
loader = model.get_train_dataloader(local_path, batch_size=32)

exec_strategy = fluid.ExecutionStrategy()
dist_strategy = fleet.DistributedStrategy()
exec_strategy.num_threads = 2
exec_strategy.num_iteration_per_drop_scope = 100
dist_strategy.execution_strategy = exec_strategy
build_strategy = fluid.BuildStrategy()
build_strategy.enable_inplace = False
build_strategy.fuse_elewise_add_act_ops = True
build_strategy.fuse_bn_act_ops = True
dist_strategy.build_strategy = build_strategy
dist_strategy.nccl_comm_num = 1

optimizer = paddle.fluid.optimizer.Momentum(
    learning_rate=configs.lr,
    momentum=configs.momentum,
    regularization=paddle.fluid.regularizer.L2Decay(0.0001))
optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
optimizer.minimize(model.loss)

trainer = X.MultiGPUTrainer()
trainer.fit(model, loader, epoch=10)
