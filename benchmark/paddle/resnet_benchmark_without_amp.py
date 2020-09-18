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
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.8"
os.environ['FLAGS_sync_nccl_allreduce'] = "1"
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
os.environ['FLAGS_cudnn_exhaustive_search'] = "1"
os.environ['FLAGS_conv_workspace_size_limit'] = "4000"
os.environ['FLAGS_cudnn_batchnorm_spatial_persistent'] = "1"
os.environ['FLAGS_fuse_parameter_memory_size'] = "16"
os.environ['FLAGS_fuse_parameter_groups_size'] = "50"

import math
import fleetx as X
import paddle.fluid as fluid
import time
import paddle.distributed.fleet as fleet
# FleetX help users to focus more on learning to train a large scale model
# if you want to learn how to write a model, FleetX is not for you
# focus more on engineering staff in fleet-x
configs = X.parse_train_configs()
fleet.init(is_collective=True)
model = X.applications.Resnet50(data_layout='NCHW')
loader = model.load_imagenet_from_file(
    "/pathto/ImageNet/train.txt", batch_size=32)
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
optimizer = fluid.optimizer.Momentum(
    learning_rate=configs.lr, momentum=configs.momentum)
optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
optimizer.minimize(model.loss)
trainer = X.MultiGPUTrainer()
trainer.fit(model, loader, epoch=10, use_dali=True)
