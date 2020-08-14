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
    
    def _dist_wrapper(generator, is_test):
        def _wrapper():
            rank = fleet.worker_index()
            nranks = fleet.worker_num()
            for idx, sample in enumerate(generator()):
                if idx % nranks == rank:
                    yield sample
        
        def _test_wrapper():
            rank = fleet.worker_index()
            nranks = fleet.worker_num()
            for idx, sample in enumerate(generator()):
                if idx // 10 % nranks == rank:
                    yield sample

        return _wrapper if not is_test else _test_wrapper

    if is_distributed:
        generator = _dist_wrapper(generator, is_test)

    loader.set_sample_generator(
            generator,
            batch_size=batch_size,
            drop_last=(not is_test),
            places=place)
    return loader

def dist_eval(exe, var_names, feed):
    prog = fluid.Program()
    with fluid.program_guard(prog):
        feed_list = []
        for name in var_names:
            feed_list.append(fluid.layers.data(
                name=name, shape=[1], dtype='float32'))
        feed_list.append(fluid.layers.data(name='length', shape=[1], dtype='int32'))

        dist_fetch = []
        for var in feed_list:
            dist_fetch.append(fluid.layers.collective._c_allreduce(
                var, reduce_type='sum', use_calc_stream=True))

        ret = exe.run(prog, feed=feed, fetch_list=dist_fetch)
    return ret

def dist_eval_ubuntu(exe, result):
    var_names = ["p_at_1_in_2", "p_at_1_in_10", "p_at_2_in_10", "p_at_5_in_10"]
    feed = {
        'p_at_1_in_2': result["1_in_2"][0],
        'p_at_1_in_10': result["1_in_10"][0],
        'p_at_2_in_10': result["2_in_10"][0],
        'p_at_5_in_10': result["5_in_10"][0],
        'length': result["1_in_2"][1],
    }

    ret = dist_eval(exe, var_names, feed)
   
    p_at_1_in_2, p_at_1_in_10, p_at_2_in_10, p_at_5_in_10, length = ret
    dist_result = {
        "1_in_2": p_at_1_in_2 / length,
        "1_in_10": p_at_1_in_10 / length,
        "2_in_10": p_at_2_in_10 / length,
        "5_in_10": p_at_5_in_10 / length
    }
    return dist_result

def dist_eval_douban(exe, result):
    var_names = ["sum_m_a_p", "sum_m_r_r", "sum_p_1", "sum_r_1", "sum_r_2", "sum_r_5"]
    feed = {
        "sum_m_a_p": result["MAP"][0],
        "sum_m_r_r": result["MRR"][0],
        "sum_p_1": result["P_1"][0],
        "sum_r_1": result["1_in_10"][0],
        "sum_r_2": result["2_in_10"][0],
        "sum_r_5": result["5_in_10"][0],
        'length': result["MAP"][1],
    }

    ret = dist_eval(exe, var_names, feed)
    sum_m_a_p, sum_m_r_r, sum_p_1, sum_r_1, sum_r_2, sum_r_5, total_num = ret

    dist_result = {
        "MAP": sum_m_a_p / total_num,
        "MRR": sum_m_r_r / total_num,
        "P_1": sum_p_1 / total_num,
        "1_in_10": sum_r_1 / total_num,
        "2_in_10": sum_r_2 / total_num,
        "5_in_10": sum_r_5 / total_num
    }
    return dist_result
