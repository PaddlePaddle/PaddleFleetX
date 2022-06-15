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
import os
import paddle.nn as nn
import time
import sys
from model import DnnModel
from reader import DnnDataset
paddle.enable_static()

def main():
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    # 模型组网
    model = DnnModel()
    model.net(is_train=True)

    # 数据加载
    data_dir = './data'
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    dataset = DnnDataset(file_list)
    loader = paddle.io.DataLoader(
        dataset,
        batch_size=1,
        places=place,
        drop_last=True)

    # 定义优化器
    optimizer = paddle.optimizer.SGD(learning_rate=0.0001)
    optimizer.minimize(model.loss)

    exe.run(paddle.static.default_startup_program())

    input_data_names = [var.name for var in model.inputs]
    epochs = 1
    print_interval = 1
    # 开始训练
    for epoch_id in range(epochs):
        for batch_id, batch in enumerate(loader()):
            fetch_batch_var = exe.run(
                program=paddle.static.default_main_program(),
                feed=dict(zip(input_data_names, batch)),
                fetch_list=[model.loss.name])
            if batch_id % print_interval == 0:
                print("[Epoch %d, batch %d] loss: %.5f" % (epoch_id, batch_id, fetch_batch_var[0]))


if __name__ == '__main__':
    main()
