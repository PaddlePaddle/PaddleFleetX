#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from ..dataset import QueueDataset, MemoryDataset
import os

class TrainerBase(object):
    def __init__(self):
        self.thread_num = 10
        self.dataset = None
        self.model = None
        self.optimizer = None

    def set_batch_size(self, batch):
        self.batch_size = batch

    def set_thread(self, thread):
        self.thread_num = thread

    def init(self, dataset=None, model=None, optimizer=None):
        if model == None or optimizer == None:
            print("Model and optimizer should be set before init")
            exit(-1)
        self.dataset = dataset
        self.optimizer = optimizer
        self.model = model
        if self.dataset:
            self.dataset.inst.set_use_var(self.model.get_input_vars())
            self.dataset.inst.set_thread(self.thread_num)
            self.dataset.inst.set_batch_size(self.batch_size)
            self.dataset.inst.set_pipe_command(self.model.get_pipe_command())
        self.optimizer.inst.minimize(model.loss)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(self.model.startup_program)

    def train_pass(self, pass_folder, **kwargs):
        raise NotImplemented("You should implement this")

    def save_inference_model(self, path):
        exe = fluid.Executor(fluid.CPUPlace())
        fluid.io.save_inference_model(
            path, [x.name for x in self.model.get_input_vars()],
            self.model.metrics.values(), exe)


class BatchTrainer(TrainerBase):
    def __init__(self):
        pass

    def run(self, data_folder):
        pass


class DistBatchTrainer(TrainerBase):
    def __init__(self):
        pass

    def run(self, data_folder):
        pass


class OnlineTrainer(TrainerBase):
    def __init__(self):
        super(OnlineTrainer, self).__init__()

    def train_pass(self, pass_folder, **kwargs):
        prefix = kwargs.get("prefix", "part")
        is_debug = kwargs.get("is_debug", False)
        exe = fluid.Executor(fluid.CPUPlace())
        files = ["{}/{}".format(pass_folder, x)
                 for x in os.listdir(pass_folder) if prefix in x]
        # train use dataset
        if self.dataset:
            self.dataset.inst.set_filelist(files)
            if isinstance(self.dataset, QueueDataset):
                exe.train_from_dataset(
                    program=self.model.main_program,
                    dataset=self.dataset.inst,
                    debug=is_debug)
            elif isinstance(self.dataset, MemoryDataset):
                self.dataset.inst.load_into_memory()
                self.dataset.inst.local_shuffle()
                exe.train_from_dataset(
                    program=self.model.main_program,
                    dataset=self.dataset.inst,
                    debug=is_debug)
        else:
            raise NotImplemented("Training with reader has"
                                 "not been implemented yet")

class DistOnlineTrainer(OnlineTrainer):
    def __init__(self):
        pass

    def init(self, dataset=None, model=None, optimizer=None):
        role_maker = PaddleCloudRoleMaker()
        fleet.init(role_maker)
        self.dist_optimizer = fleet.distributed_optimizer(self.optimizer.inst)
        self.dist_optimizer.minimize(self.model.loss)
        if fleet.is_server():
            fleet.init_server()
            fleet.run_server()
        elif fleet.is_worker():
            fleet.init_worker()

    def train_pass(self, pass_folder, **kwargs):
        exe = fluid.Executor(fluid.CPUPlace())

    
