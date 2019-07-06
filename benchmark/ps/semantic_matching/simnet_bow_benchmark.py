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

import os
import sys
import paddle.fluid as fluid
from nets import bow_encoder
base_lr = 0.0001
batch_size = 128
emb_lr = 5.0 * batch_size
fc_lr = 200.0
dict_dim = 1451594
emb_dim = 128
hid_dim = 128
margin = 0.1


q = fluid.layers.data(
    name="query", shape=[1], dtype="int64", lod_level=1)
pt = fluid.layers.data(
    name="pos_title", shape=[1], dtype="int64", lod_level=1)
nt = fluid.layers.data(
    name="neg_title", shape=[1], dtype="int64", lod_level=1)

avg_cost, pt_s, nt_s, pnum, nnum, train_pn = \
        bow_encoder(q, pt, nt, dict_dim, emb_dim,
                    hid_dim, emb_lr, fc_lr, margin)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=base_lr)
sgd_optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


dataset = fluid.DatasetFactory().create_dataset()
dataset.set_use_var([q, pt, nt])
pipe_command = 'python reader.py'
dataset.set_pipe_command(pipe_command)

batches = [32, 64, 128, 256, 512, 1024]
threads = [10, 20]

filelist = ["train_raw/part-00000.raw"]

time_summary = {}
import time
for bs in batches:
    for thr in threads:
        start_time = time.time()
        dataset.set_batch_size(bs)
        dataset.set_thread(thr)
        dataset.set_filelist(filelist * thr)
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset,
                               fetch_list=[train_pn],
                               fetch_info=["pos/neg"],
                               print_period=10000,
                               debug=False)
        end_time = time.time()
        time_summary[(bs, thr)] = end_time - start_time

print("batch v.s threads\tthread=%d\tthread=%d" % (threads[0], threads[1]))
total_pair = 1584030.0 * thr
for bs in batches:
    out_str = "batch=%d" % bs
    for thr in threads:
        out_str += "\t%7.4f/s" % (total_pair / time_summary[(bs, thr)])
    print(out_str)

if __name__ == "__main__":
    train()
