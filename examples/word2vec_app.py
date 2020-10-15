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

paddle.enable_static()

role = fleet.PaddleCloudRoleMaker()
fleet.init(role)

model = X.applications.Word2vec()

dist_strategy = fleet.DistributedStrategy()
dist_strategy.a_sync = True

optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
else:
    """
    need config loader correctly.
    """
    
    local_path = downloader.download_from_hdfs(
                     'w2v_train_data.yaml', local_path="train_data",
                     shard_num=fleet.worker_num(),
                     shard_id=fleet.worker_index())
    
    thirdparty_path = downloader.download_from_hdfs(
                     'w2v_thirdparty.yaml', local_path="thirdparty")
    
    train_file_list=[str(local_path) + "/%s" % x
                     for x in os.listdir(local_path)]
    
    dataset = model.load_dataset_from_file(dict_path="./thirdparty/test_build_dict", file_list=train_file_list)

    trainer = X.CPUDatasetTrainer(calc_line=False)
    trainer.fit(model, dataset, epoch=10)
