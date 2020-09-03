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
import paddle.distributed.fleet as fleet

configs = X.parse_train_configs()
model = X.applications.BertBase()
wiki_downloader = X.utils.WikiDataDownloader()
local_path = wiki_downloader.download_from_bos(local_path='./data')

loader = model.load_digital_dataset_from_file(
    data_dir='{}/train_data'.format(local_path),
    vocab_path='{}/vocab.txt'.format(local_path))

fleet.init(is_collective=True)
dist_strategy = fleet.DistributedStrategy()
dist_strategy.amp = True

learning_rate = X.utils.linear_warmup_decay(configs.lr, 4000, 1000000)
clip = paddle.fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
optimizer = paddle.fluid.optimizer.Adam(
    learning_rate=learning_rate, grad_clip=clip)
optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
optimizer.minimize(model.loss)

trainer = X.MultiGPUTrainer()
trainer.fit(model, loader, epoch=10)
