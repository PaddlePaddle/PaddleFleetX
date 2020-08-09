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
from dataloader import sample_generator
from paddle.fluid.incubate.fleet.collective import fleet
import os

def create_dataloader(feed_var_list, filelist,
        place, batch_size, dict_path, max_turn_num,
        max_turn_len, is_test, data_source, is_distributed):
    loader = fluid.io.DataLoader.from_generator(
            feed_list=feed_var_list,
            capacity=8,
            iterable=True)
    data_conf = {
        "max_turn_num": max_turn_num,
        "max_turn_len": max_turn_len,
    }
    generator = sample_generator(
            filelist,
            dict_path,
            data_conf,
            data_source)
    
    def _dist_wrapper(generator):
        def _wrapper():
            rank = fleet.worker_index()
            nranks = fleet.worker_num()
            for idx, sample in enumerate(generator()):
                if idx % nranks == rank:
                    yield sample
        return _wrapper

    if is_distributed:
        generator = _dist_wrapper(generator)

    loader.set_sample_generator(
            generator,
            batch_size=batch_size,
            drop_last=(not is_test),
            places=place)
    return loader
