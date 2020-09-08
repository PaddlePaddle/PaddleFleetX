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

os.environ['FLAGS_enable_parallel_graph'] = "0"
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.98"
os.environ['FLAGS_sync_nccl_allreduce'] = "1"
os.environ['FLAGS_eager_delete_tensor_gb'] = "0"
os.environ['FLAGS_fuse_parameter_memory_size'] = "32"
os.environ['FLAGS_fuse_parameter_groups_size'] = "50"

import fleetx as X
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
# FleetX help users to focus more on learning to train a large scale model
# if you want to learn how to write a model, fleetx is not for you
# focus more on engineering staff in fleet-x

configs = X.parse_train_configs()
fleet.init(is_collective=True)
# load BertLarge / BertBase model
model = X.applications.BertLarge()
#model = X.applications.BertBase()

data_loader = model.load_digital_dataset_from_file(
    data_dir='./train_data', vocab_path='./vocab.txt')

learning_rate = X.utils.linear_warmup_decay(configs.lr, 4000, 1000000)
exec_strategy = fluid.ExecutionStrategy()
exec_strategy.num_threads = 2
exec_strategy.num_iteration_per_drop_scope = 1
dist_strategy = fleet.DistributedStrategy()
dist_strategy.execution_strategy = exec_strategy
dist_strategy.nccl_comm_num = 3
optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)

trainer = X.MultiGPUTrainer()

trainer.fit(model, data_loader, 10)
