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

os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.98"
os.environ['FLAGS_sync_nccl_allreduce'] = "1"
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
os.environ['FLAGS_cudnn_exhaustive_search'] = "1"
os.environ['FLAGS_fuse_parameter_memory_size'] = "50"
os.environ['FLAGS_fuse_parameter_groups_size'] = "50"

import fleetx as X
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
# FleetX help users to focus more on learning to train a large scale model
# if you want to learn how to write a model, FleetX is not for you
# focus more on engineering staff in fleet-x
configs = X.parse_train_configs()
fleet.init(is_collective=True)
model = X.applications.Transformer()

data_loader = model.load_wmt16_dataset_from_file(
    '/pathto/wmt16_ende_data_bpe/vocab_all.bpe.32000',
    '/pathto/wmt16_ende_data_bpe/vocab_all.bpe.32000',
    '/pathto/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de')

optimizer = fluid.optimizer.Adam(
    learning_rate=configs.lr,
    beta1=configs.beta1,
    beta2=configs.beta2,
    epsilon=configs.epsilon)
dist_strategy = fleet.DistributedStrategy()
optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(model.loss)

trainer = X.MultiGPUTrainer()
trainer.fit(model, data_loader, epoch=10)
