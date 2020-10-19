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
import numpy as np
import paddle.distributed.fleet as fleet


class Trainer(object):
    def __init__(self):
        """
        
        """
        self.place = None


class CPUTrainer(Trainer):
    def __init__(self):
        super(CPUTrainer, self).__init__()
        self.place = fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)

    def fit(self, model, dataloader, epoch, start_step=10):
        fleet.init_worker()
        self.exe.run(fluid.default_startup_program())

        for epoch_id in range(epoch):
            total_time = 0
            step = 0
            for data in dataloader():
                if step > start_step:
                    start_time = time.time()
                loss = self.exe.run(fluid.default_main_program(),
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
        self.place = fluid.CUDAPlace(
            int(os.environ.get('FLAGS_selected_gpus', 0)))
        self.exe = fluid.Executor(self.place)
        self.exe.run(fluid.default_startup_program())

    def fit(self, model, dataloader, epoch, use_dali=False, start_step=10):

        for epoch_id in range(epoch):
            total_time = 0
            step = 0
            for data in dataloader:
                if step > start_step:
                    start_time = time.time()
                loss = self.exe.run(fluid.default_main_program(),
                                    feed=data,
                                    fetch_list=[model.loss.name],
                                    use_program_cache=True)
                if step > start_step:
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    print(
                        "epoch id: %d, step%d, train_loss: %f, total time cost = %f, step per second: %f, speed: %f"
                        % (epoch_id, step, loss[0], total_time,
                           (step - start_step) / total_time,
                           1 / (end_time - start_time)))
                step += 1
            if use_dali:
                dataloader.reset()

    def val(self,
            model,
            dataloader,
            target_list,
            current_epoch=-1,
            use_dali=False):
        self.test_program = model.main_prog.clone(for_test=True)
        fetch_target = []
        results = {}
        for item in target_list:
            if item in model.target.keys():
                fetch_target.append(model.target[item].name)
                results[item] = []
            else:
                raise Exception("ERROR: Current model only support target: {}".
                                format(model.target.keys()))

        for data in dataloader:
            result = self.exe.run(self.test_program,
                                  feed=data,
                                  fetch_list=fetch_target,
                                  use_program_cache=True)
            for item in target_list:
                results[item].append(np.mean(result[target_list.index(item)]))

        log_info = ""
        for item in target_list:
            log_info += ", {} = {}".format(item, np.mean(results[item]))
        if current_epoch > 0:
            print("Test Epoch {}{}".format(current_epoch, log_info))
        else:
            print("Test Result {}".format(log_info))
        if use_dali:
            dataloader.reset()

    def quick_benchmark(self,
                        model,
                        dataloader,
                        start_step=20,
                        end_step=200):
        step = 0
        total_time = 0
        total_step = 0
        counting_time = False
        for data in dataloader:
            if step > start_step and step <= end_step:
                start_time = time.time()
            loss = self.exe.run(fluid.default_main_program(),
                                feed=data,
                                fetch_list=[],
                                use_program_cache=True)
            if step > start_step and step <= end_step:
                end_time = time.time()
                total_time += (end_time - start_time)

            if step > end_step:
                break

            step += 1

        mean_qps = (end_step - start_step) / total_time
        return mean_qps

    def benchmark(self,
                  model,
                  dataloader,
                  epoch,
                  use_dali=False,
                  start_step=20):
        for epoch_id in range(epoch):
            total_time = 0
            step = 0
            for data in dataloader:
                if step > start_step and step <= start_step + 100:
                    start_time = time.time()
                loss = self.exe.run(fluid.default_main_program(),
                                    feed=data,
                                    fetch_list=[model.loss.name],
                                    use_program_cache=True)
                if step > start_step and step <= start_step + 100:
                    end_time = time.time()
                    total_time += (end_time - start_time)
                step += 1
            average_speed = 100 / total_time
            if use_dali:
                dataloader.reset()
        return average_speed

    def benchmark_val(self, model, dataloader, target_list, use_dali=False):
        self.test_program = model.main_prog.clone(for_test=True)
        fetch_target = []
        results = {}
        for item in target_list:
            if item in model.target.keys():
                fetch_target.append(model.target[item].name)
                results[item] = []
            else:
                raise Exception("ERROR: Current model only support target: {}".
                                format(model.target.keys()))

        for data in dataloader:
            result = self.exe.run(self.test_program,
                                  feed=data,
                                  fetch_list=fetch_target,
                                  use_program_cache=True)
            for item in target_list:
                results[item].append(np.mean(result[target_list.index(item)]))

        if use_dali:
            dataloader.reset()
        return results
