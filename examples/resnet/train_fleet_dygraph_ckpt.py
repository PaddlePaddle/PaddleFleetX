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
from resnet_dygraph import ResNet

import os

base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4

epoch = 20
batch_size = 32
class_dim = 102

checkpoint_path = '/checkpoint/resnet.chkpt'


def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label

    return __reader__

def optimizer_setting(parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=base_lr,
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay),
        parameters=parameter_list)
    return optimizer


def train_resnet():
    #paddle.set_device('cpu')
    # paddle.set_device('gpu:0')
    fleet.init(is_collective=True)

    resnet = ResNet(class_dim=class_dim, layers=50)

    ## recover from checkpoint
    if int(fleet.local_rank()) == 0 and checkpoint_path and os.path.isfile(checkpoint_path):
        print("try to load checkpoint...")
        try:
            chkpt = paddle.load(checkpoint_path)
            resnet.set_state_dict(chkpt)
            start_epoch = chkpt.get('epoch',0)
            print("load checkpoint succuss for epoch", start_epoch)
        except Exception as e:
            print("load checkpoint failed", e)

    optimizer = optimizer_setting(parameter_list=resnet.parameters())
    optimizer = fleet.distributed_optimizer(optimizer)
    resnet = fleet.distributed_model(resnet)

    train_reader = paddle.batch(
            reader_decorator(paddle.dataset.flowers.train(use_xmap=True)),
            batch_size=batch_size,
            drop_last=True)

    train_loader = paddle.io.DataLoader.from_generator(
        capacity=32,
        use_double_buffer=True,
        iterable=True,
        return_list=True)
    train_loader.set_sample_list_generator(train_reader)

    # keep going with previous epoch
    for eop in range(epoch):
        resnet.train()

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True

            out = resnet(img)
            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            avg_loss = paddle.mean(x=loss)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

            dy_out = avg_loss.numpy()

            avg_loss.backward()

            optimizer.minimize(avg_loss)
            resnet.clear_gradients()
            if batch_id % 10 == 0:
                # save checkpoint in rank 0
                if int(fleet.local_rank()) == 0:
                    state_dict = resnet.state_dict()
                    # add user defined data
                    state_dict['epoch'] = eop
                    paddle.save(state_dict, checkpoint_path)
                print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, dy_out, acc_top1, acc_top5))

if __name__ == '__main__':
    train_resnet()
