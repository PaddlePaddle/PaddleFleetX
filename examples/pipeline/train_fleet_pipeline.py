# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import argparse
import ast
import paddle
from paddle.distributed import fleet
import resnet_pipeline as resnet
import os

base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4

epoch = 10
batch_size = 32
class_dim = 1000

def optimizer_setting(parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=base_lr,
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay),
        parameters=parameter_list)
    return optimizer

def get_train_loader(feed_list, place):
    def reader_decorator(reader):
        def __reader__():
            for item in reader():
                img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield img, label

        return __reader__

    train_reader = paddle.batch(
        reader_decorator(paddle.dataset.flowers.train(use_xmap=False)),
        batch_size=batch_size,
        drop_last=True)
    train_loader = paddle.io.DataLoader.from_generator(
        capacity=32,
        use_double_buffer=True,
        feed_list=feed_list,
        iterable=False)
    train_loader.set_sample_list_generator(train_reader, place)
    return train_loader

def train_resnet():
    paddle.enable_static()
    paddle.vision.set_image_backend('cv2')

    place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    with paddle.fluid.device_guard("gpu:0"):
        image = paddle.static.data(name="x", shape=[None, 3, 224, 224], dtype='float32')
        label= paddle.static.data(name="y", shape=[None, 1], dtype='int64')
        train_loader = get_train_loader([image, label], place)

        model = resnet.ResNet(layers=50)
        out, offset = model.net(input=image, class_dim=class_dim)

    with paddle.fluid.device_guard("gpu:%d"%offset):
        avg_cost = paddle.nn.functional.cross_entropy(input=out, label=label)
        acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
        acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

    strategy = fleet.DistributedStrategy()
    strategy.pipeline = True
    strategy.pipeline_configs = {"accumulate_steps": 8,
                                 "micro_batch_size": batch_size}
    fleet.init(is_collective=True, strategy=strategy)
    optimizer = optimizer_setting()
    optimizer = fleet.distributed_optimizer(optimizer)
    optimizer.minimize(avg_cost)

    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    epoch = 10
    step = 0
    train_loader.start()
    for eop in range(epoch):
        batch_id = 0
        while True:
            try:
                if fleet.worker_index() == 3:
                    loss, acc1, acc5 = exe.run(paddle.static.default_main_program(), fetch_list=[avg_cost, acc_top1, acc_top5])             
                else:
                    exe.run(paddle.static.default_main_program())             
                batch_id += 1
                if batch_id % 5 == 0 and fleet.worker_index() == 3:
                    print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, loss, acc1, acc5))
            except paddle.fluid.core.EOFException:
                train_loader.reset()
                break

if __name__ == '__main__':
    train_resnet()
