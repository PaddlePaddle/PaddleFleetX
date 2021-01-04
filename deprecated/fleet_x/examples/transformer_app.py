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
import paddle
import fleetx as X
import paddle.distributed.fleet as fleet


paddle.enable_static()
configs = X.parse_train_configs()
fleet.init(is_collective=True)
model = X.applications.Transformer()
wmt_downloader = X.utils.Downloader()
local_path = wmt_downloader.download_from_bos(
    fs_yaml='https://fleet.bj.bcebos.com/small_datasets/yaml_example/wmt.yaml',
    local_path='./data')
data_loader = model.get_train_dataloader(local_path)

optimizer = paddle.fluid.optimizer.Adam(
    learning_rate=configs.lr,
    beta1=configs.beta1,
    beta2=configs.beta2,
    epsilon=configs.epsilon)
dist_strategy = fleet.DistributedStrategy()
optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(model.loss)

trainer = X.MultiGPUTrainer()
trainer.fit(model, data_loader, epoch=1)
