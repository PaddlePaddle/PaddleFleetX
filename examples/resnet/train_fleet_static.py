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
import paddle
from paddle.distributed import fleet
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock
from paddle.io import Dataset, BatchSampler, DataLoader
import os

base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4

epoch = 10
batch_num = 100
batch_size = 32
class_dim = 102

def optimizer_setting(parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=base_lr,
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay),
        parameters=parameter_list)
    return optimizer


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([3, 224, 224]).astype('float32')
        label = np.random.randint(0, class_dim - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

def get_train_loader(place):
    dataset = RandomDataset(batch_num * batch_size)
    train_loader = DataLoader(dataset,
                 places=place, 
                 batch_size=batch_size,
                 shuffle=True,
                 drop_last=True,
                 num_workers=2)    
    return train_loader

def train_resnet():
    paddle.enable_static()
    paddle.vision.set_image_backend('cv2')

    image = paddle.static.data(name="x", shape=[None, 3, 224, 224], dtype='float32')
    label= paddle.static.data(name="y", shape=[None, 1], dtype='int64')

    model = ResNet(BottleneckBlock, 50, num_classes=class_dim)
    out = model(image)
    avg_cost = paddle.nn.functional.cross_entropy(input=out, label=label)
    acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

    place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    
    train_loader = get_train_loader(place)

    strategy = fleet.DistributedStrategy()
    fleet.init(is_collective=True, strategy=strategy)
    optimizer = optimizer_setting()
    optimizer = fleet.distributed_optimizer(optimizer)
    optimizer.minimize(avg_cost)

    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    epoch = 10
    step = 0
    for eop in range(epoch):
        for batch_id, (image, label) in enumerate(train_loader()):
            loss, acc1, acc5 = exe.run(paddle.static.default_main_program(), feed={'x': image, 'y': label}, fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])             
            if batch_id % 5 == 0:
                print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, loss, acc1, acc5))

if __name__ == '__main__':
    train_resnet()
