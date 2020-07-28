#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import os

def load_filelist(filelist_path, is_distributed):
    if not os.path.exists(filelist_path):
        raise SystemExit("{} not exists.".format(filelist_path))

    files = []
    with open(filelist_path) as fs:
        for idx, line in enumerate(fs):
            line = line.strip()
            if is_distributed:
                rank = fleet.worker_index()
                nranks = fleet.worker_num()
                if idx % nranks == rank:
                    files.append(line)
            else:
                files.append(line)
    return files

def create_dataset(feed_var_list, filelist, batch_size, 
        thread_num, dict_path, max_turn_num, max_turn_len, 
        data_source):
    dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
    dataset.set_batch_size(batch_size)
    dataset.set_filelist(filelist)
    dataset.set_use_var(feed_var_list)
    pipe_command = "python data_generator.py {} {} {} {}".format(
            dict_path, max_turn_num, max_turn_len, data_source)
    dataset.set_pipe_command(pipe_command)
    return dataset

def create_dataloader(feed_var_list, filelist, place, batch_size, thread_num,
        dict_path, max_turn_num, max_turn_len, is_test, data_source):
    dataset = create_dataset(feed_var_list, filelist,
            batch_size, thread_num, dict_path, max_turn_num,
            max_turn_len, data_source)
    loader = fluid.io.DataLoader.from_dataset(dataset, place, drop_last=(not is_test))
    return loader
