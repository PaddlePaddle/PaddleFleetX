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
from ..utils import hdfs_ls
import sys
import os

class InferBase(object):
    def __init__(self):
        self.thread_num = 10
        self.dataset = None
        self.model = None

    def set_batch_size(self, batch):
        self.batch_size = batch

    def set_thread(self, thread):
        self.thread_num = thread

    def init(self, dataset=None, model=None):
        if model == None:
            print("Model should be set before init")
            exit(-1)
        self.dataset = dataset
        self.model = model
        if self.dataset:
            self.dataset.inst.set_use_var(self.model.get_infer_input_vars())
            self.dataset.inst.set_thread(self.thread_num)
            self.dataset.inst.set_batch_size(self.batch_size)
            self.dataset.inst.set_pipe_command(self.model.get_infer_pipe_command())

    def infer_pass(self, pass_folder, **kwargs):
        raise NotImplemented("")

    def load_inference_model(self, 
                             local_input=None,
                             remote_input=None,
                             local_output=None):
        if local_path != None:
            local_output = local_path

        if remote_path != None:
            if local_output == None:
                local_output = "./current_model"
            hdfs_get(remote_path, local_output, self.fs_name, self.ugi)
        [self.infer_program, self.feed_target_names, fetch_targets] = \
                        fluid.io.load_inference_model(local_output, exe)


class BatchInfer(InferBase):
    def __init__(self):
        pass


class DistBatchTrainer(InferBase):
    def __init__(self):
        pass

class OnlineInfer(InferBase):
    def __init__(self):
        super(OnlineInfer, self).__init__()

    def infer_pass(self, pass_folder, **kwargs):
        pass


class DistOnlineInfer(OnlineInfer):
    def __init__(self):
        super(DistOnlineInfer, self).__init__()

    def init(self, dataset=None, model=None):
        if model == None:
            print("Model should be ")

    def infer_pass(self, pass_folder, **kwargs):
        prefix = kwargs.get("prefix", "part")
        is_debug = kwargs.get("is_debug", False)
        handler = kwargs.get("handler", None)
        exe = fluid.Executor(fluid.CPUPlace())
        filelist = hdfs_ls([pass_folder],
                           self.fs_name,
                           self.ugi)
        def has_prefix(x):
            return prefix in x
        filelist = filter(has_prefix, filelist)
        my_filelist = ps.fleet.split_files(filelist)
        self.dataset.inst.set_filelist(my_filelist)
        self.dataset.inst.load_into_memory()
        exe.infer_from_dataset(
            program=self.model.infer_program,
            dataset=self.dataset.inst,
            debug=is_debug,
            fetch_handler=handler)
        
