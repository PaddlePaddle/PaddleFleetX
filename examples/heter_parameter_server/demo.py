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

import random
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet

paddle.enable_static()


def sample_generator_creator():
    def __reader__():
        for i in range(100):
            fake_input = random.randint(0, 1000-1)
            fake_label = random.randint(0, 1)
            yield fake_input, fake_label

    return __reader__


with fluid.device_guard("cpu"):
    input_data = paddle.static.data(name="sparse_input", shape=[
        None, 1], dtype="int64")
    input_label = paddle.static.data(
        name="label", shape=[None, 1], dtype="int64")
    label = paddle.cast(input_label, dtype="float32")
    embedding = paddle.static.nn.embedding(
        input_data, is_sparse=True, size=[1000, 128])


with fluid.device_guard("gpu"):
    fc1 = paddle.static.nn.fc(embedding, size=1024, activation="relu")
    fc2 = paddle.static.nn.fc(fc1, size=512, activation="relu")
    fc3 = paddle.static.nn.fc(fc2, size=256, activation="relu")
    predict = paddle.static.nn.fc(fc3, size=2, activation="softmax")
    label = paddle.cast(label, dtype="int64")
    cost = paddle.nn.functional.cross_entropy(input=predict, label=label)
    paddle.static.Print(cost, message="heter_cost")

fleet.init()
strategy = fleet.DistributedStrategy()
strategy.a_sync = True
strategy.a_sync_configs = {"heter_worker_device_guard": "gpu"}

optimizer = paddle.optimizer.Adam(1e-4)
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(cost)

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
elif fleet.is_worker():
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    fleet.init_worker()

    loader = fluid.io.DataLoader.from_generator(
        feed_list=[input_data, input_label], capacity=16)
    loader.set_sample_generator(
        sample_generator_creator(), batch_size=10, drop_last=True, places=place)

    for data in loader():
        exe.run(fluid.default_main_program(), feed=data)

    fleet.stop_worker()
    print("heter parameter server run success")
