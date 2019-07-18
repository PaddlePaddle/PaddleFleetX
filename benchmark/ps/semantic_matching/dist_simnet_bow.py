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

import argparse
import os
import sys
import logging
import time
import args as arg

import paddle.fluid as fluid
from nets import bow_encoder

import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

base_lr = 0.0001
batch_size = 128
emb_lr = 5.0 * batch_size
fc_lr = 200.0
dict_dim = 1451594
emb_dim = 128
hid_dim = 128
margin = 0.1

args = arg.parse_args()

q = fluid.layers.data(
    name="query", shape=[1], dtype="int64", lod_level=1)
pt = fluid.layers.data(
    name="pos_title", shape=[1], dtype="int64", lod_level=1)
nt = fluid.layers.data(
    name="neg_title", shape=[1], dtype="int64", lod_level=1)

avg_cost, pt_s, nt_s, pnum, nnum, train_pn = \
        bow_encoder(q, pt, nt, dict_dim, emb_dim,
                    hid_dim, emb_lr, fc_lr, margin)

exe = fluid.Executor(fluid.CPUPlace())

role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)

strategy = DistributeTranspilerConfig()
strategy.sync_mode = False

optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(avg_cost)

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()

elif fleet.is_worker():
    fleet.init_worker()
    exe.run(fleet.startup_program)

    thread_num = 12
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_batch_size(batch_size)
    use_vars = [q, pt, nt]
    use_var_names = [var.name for var in use_vars]
    dataset.set_use_var(use_vars)
    dataset.set_batch_size(batch_size)
    pipe_command = 'python reader.py'
    dataset.set_pipe_command(pipe_command)
    filelist = ["train_raw/%s" % x for x in os.listdir("train_raw")]

    dataset.set_filelist(filelist)
    dataset.set_thread(thread_num)

    epochs = 40
    save_dirname = "simnet_bow_model"

    for i in range(epochs):
        time_start = time.time()
        exe.train_from_dataset(program=fleet.main_program,
                               dataset=dataset,
                               debug=True)
        sys.stderr.write("epoch %d finished, use time=%d\n" % ((i + 1), time.time() - time_start))

        if fleet.worker_index() == 0:
            model_path = "%s/epoch%d.model" % (save_dirname, (i + 1))
            fluid.io.save_inference_model(model_path, use_var_names, [pnum, nnum], exe)
    fleet.stop_worker()

