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
import time
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet


class Trainer(object):
    def __init__(self):
        self.place = None


class CPUTrainer(Trainer):
    def __init__(self):
        super(CPUTrainer, self).__init__()

    def fit(self, model, dataloader, epoch, start_step=10):
        fleet.init_worker()
        self.place = fluid.CPUPlace()
        exe = fluid.Executor(self.place)
        exe.run(fluid.default_startup_program())

        for epoch_id in range(epoch):
            total_time = 0
            step = 0
            for data in dataloader:
                if step > start_step:
                    start_time = time.time()
                loss = exe.run(fluid.default_main_program(),
                               feed=data,
                               fetch_list=[model.loss.name])
                if step > start_step:
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    print(
                        "worker_index: %d, step%d, train_loss: %f, total time cost = %f, step per second: %f, speed: %f"
                        % (fleet.worker_index(), step, loss[0], total_time,
                           (step - start_step) / total_time,
                           1 / (end_time - start_time)))
                step += 1

        fleet.stop_worker()


class MultiGPUTrainer(Trainer):
    def __init__(self):
        super(MultiGPUTrainer, self).__init__()

    def fit(self, model, dataloader, epoch, use_dali=False, start_step=10):
        self.place = fluid.CUDAPlace(
            int(os.environ.get('FLAGS_selected_gpus', 0)))
        exe = fluid.Executor(self.place)
        exe.run(fluid.default_startup_program())

        for epoch_id in range(epoch):
            total_time = 0
            step = 0
            for data in dataloader:
                if step > start_step:
                    start_time = time.time()
                loss = exe.run(fluid.default_main_program(),
                               feed=data,
                               fetch_list=[model.loss.name])
                if step > start_step:
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    print(
                        "worker_index: %d, step%d, train_loss: %f, total time cost = %f, step per second: %f, speed: %f"
                        % (fleet.worker_index(), step, loss[0], total_time,
                           (step - start_step) / total_time,
                           1 / (end_time - start_time)))
                step += 1
            if use_dali:
                dataloader.reset()
