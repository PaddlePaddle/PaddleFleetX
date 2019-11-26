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
from ..utils import hdfs_ls, hdfs_rmr, hdfs_put
import sys
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler as ps
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
import logging
logging.basicConfig()

import os

class TrainerBase(object):
    def __init__(self):
        self.thread_num = 10
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.logger = logging.getLogger("TrainerBase")

    def print_system_info(self):
        sys_info = get_system_info()
        for key in sys_info:
            print("{}: {}".format(key, sys_info[key]))

    def set_batch_size(self, batch):
        self.batch_size = batch

    def set_thread(self, thread):
        self.thread_num = thread

    def set_dfs_config(self, fs_name, ugi):
        self.fs_name = fs_name
        self.ugi = ugi

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

    def save_inference_model(self, local_path, remote_path=None):
        if remote_path == None:
            exe = fluid.Executor(fluid.CPUPlace())
            fluid.io.save_inference_model(
                local_path, [x.name for x in self.model.get_input_vars()],
                self.model.metrics.values(), exe)


class BatchTrainer(TrainerBase):
    def __init__(self):
        pass


class DistBatchTrainer(TrainerBase):
    def __init__(self):
        pass


class OnlineTrainer(TrainerBase):
    def __init__(self):
        super(OnlineTrainer, self).__init__()

    def train_pass(self, pass_folder, **kwargs):
        prefix = kwargs.get("prefix", "part")
        is_debug = kwargs.get("is_debug", False)
        handler = kwargs.get("handler", None)
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
                    debug=is_debug,
                    fetch_handler=handler)
            elif isinstance(self.dataset, MemoryDataset):
                self.dataset.inst.load_into_memory()
                self.dataset.inst.local_shuffle()
                exe.train_from_dataset(
                    program=self.model.main_program,
                    dataset=self.dataset.inst,
                    debug=is_debug,
                    fetch_handler=handler)
        else:
            raise NotImplemented("Training with reader has"
                                 "not been implemented yet")

class DistOnlineTrainer(OnlineTrainer):
    def __init__(self):
        super(DistOnlineTrainer, self).__init__()

    def init(self, dataset=None, model=None, optimizer=None):
        if model == None or optimizer == None:
            print("Model and optimizer should be set before init")
            exit(-1)
        self.dataset = dataset
        self.optimizer = optimizer
        self.model = model
        role = role_maker.MPISymetricRoleMaker()
        ps.fleet.init(role)
        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = False
        self.dist_optimizer = ps.fleet.distributed_optimizer(
            self.optimizer.inst, strategy)
        self.dist_optimizer.minimize(self.model.loss)
        if ps.fleet.is_server():
            ps.fleet.init_server()
            ps.fleet.run_server()
        elif ps.fleet.is_worker():
            self.logger.info("worker index {}".format(ps.fleet.worker_index()))
            import time
            time.sleep(5)
            ps.fleet.init_worker()
            if self.dataset:
                self.dataset.inst.set_use_var(self.model.get_input_vars())
                self.dataset.inst.set_thread(self.thread_num)
                self.dataset.inst.set_batch_size(self.batch_size)
                self.dataset.inst.set_pipe_command(self.model.get_pipe_command())
                self.dataset.inst.set_hdfs_config(self.fs_name, self.ugi)
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(ps.fleet.startup_program)

    def train_pass(self, pass_folder, **kwargs):
        # global_shuffle, preload, prefix, is_debug
        prefix = kwargs.get("prefix", "part")
        is_debug = kwargs.get("is_debug", False)
        handler = kwargs.get("handler", None)
        exe = fluid.Executor(fluid.CPUPlace())
        filelist = hdfs_ls([pass_folder], self.fs_name, self.ugi)
        def has_prefix(x):
            return prefix in x
        filelist = filter(has_prefix, filelist)
        my_filelist = ps.fleet.split_files(filelist)
        self.dataset.inst.set_filelist(my_filelist)
        self.dataset.inst.load_into_memory()
        self.dataset.inst.local_shuffle()
        exe.train_from_dataset(
            program=ps.fleet.main_program,
            dataset=self.dataset.inst,
            debug=is_debug,
            fetch_handler=handler)
        sys.stdout.write("current pass has {} files".format(len(filelist)))
        sys.stdout.write("worker index {} has {} files\n".format(
            ps.fleet.worker_index(), len(my_filelist)))
        sys.stdout.write("going to barrier worker current current pass\n")
        sys.stdout.flush()
        ps.fleet._role_maker._barrier_worker()
        sys.stdout.write("barrier worker done.\n")
        sys.stdout.flush()

    def save_inference_model(self, local_path, remote_path=None):
        exe = fluid.Executor(fluid.CPUPlace())
        fluid.io.save_inference_model(
            local_path, [x.name for x in self.model.get_input_vars()],
            self.model.metrics.values(), exe)
        if remote_path == None:
            sys.stdout.write("WARNING: You should assign hdfs path to save model")
        else:
            if ps.fleet.worker_index() == 0:
                hdfs_rmr(remote_path, self.fs_name, self.ugi)
                hdfs_put(local_path, remote_path, self.fs_name, self.ugi)
