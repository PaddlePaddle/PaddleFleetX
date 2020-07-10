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
import math
import os
import fleet_lightning as lighting
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import time
# lightning help users to focus more on learning to train a large scale model
# if you want to learn how to write a model, lightning is not for you
# focus more on engineering staff in fleet-lightning

configs = lighting.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)

model = lighting.applications.Resnet50()

loader = model.load_imagenet_from_file("/pathto/ImageNet/train.txt")
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.8"
os.environ['FLAGS_sync_nccl_allreduce'] = "1"
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
os.environ['FLAGS_cudnn_exhaustive_search'] = "1"
os.environ['FLAGS_conv_workspace_size_limit'] = "4000"
os.environ['FLAGS_cudnn_batchnorm_spatial_persistent'] = "1"
os.environ['FLAGS_fuse_parameter_memory_size'] = "16"
os.environ['FLAGS_fuse_parameter_groups_size'] = "50"

exec_strategy = fluid.ExecutionStrategy()
dist_strategy = DistributedStrategy()
exec_strategy.num_threads = 2
exec_strategy.num_iteration_per_drop_scope = 100
dist_strategy.exec_strategy = exec_strategy
dist_strategy.enable_inplace = False
dist_strategy.nccl_comm_num = 1
dist_strategy.fuse_elewise_add_act_ops = True
dist_strategy.fuse_bn_act_ops = True

optimizer = fluid.optimizer.Momentum(
    learning_rate=configs.lr,
    momentum=configs.momentum,
    regularization=fluid.regularizer.L2Decay(0.0001))
optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
optimizer.minimize(model.loss)

place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

for epoch_id in range(2):
    total_time = 0
    for i, data in enumerate(loader()):
        if i >= 100:
            start_time = time.time()
        cost_val = exe.run(fleet.main_program,
                           feed=data,
                           fetch_list=[model.loss.name])
        if i >= 100:
            end_time = time.time()
            total_time += (end_time - start_time)
            print(
                "worker_index: %d, step%d cost = %f, total time cost = %f, average speed = %f, speed = %f"
                % (fleet.worker_index(), i, cost_val[0], total_time,
                   (i - 99) / total_time, 1 / (end_time - start_time)))
