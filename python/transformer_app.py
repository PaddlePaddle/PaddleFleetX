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
import fleet_lightning as lighting
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import time
import numpy as np
# lightning help users to focus more on learning to train a large scale model
# if you want to learn how to write a model, lightning is not for you
# focus more on engineering staff in fleet-lightning
configs = lighting.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)
model = lighting.applications.Transformer()
place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
data_reader = model.load_wmt16_dataset_from_file(
    '/ssd1/jingqinghe/gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000',
    '/ssd1/jingqinghe/gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000',
    '/ssd1/jingqinghe/gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de'
)
optimizer = fluid.optimizer.Adam(
    learning_rate=configs.lr,
    beta1=configs.beta1,
    beta2=configs.beta2,
    epsilon=configs.epsilon)
optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(model.loss)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
for i in range(2):
    while True:
        try:
            feed_dict_list = model.generate_feed_dict_list(data_reader())
            out = exe.run(program=fleet.main_program,
                          fetch_list=model.loss,
                          feed=feed_dict_list)
            print("cost = %f" % out[0])
        except (StopIteration, fluid.core.EOFException):
            break
