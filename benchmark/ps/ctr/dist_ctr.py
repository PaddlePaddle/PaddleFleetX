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

import paddle.fluid as fluid

from args import parse_args

from network_conf import ctr_dnn_model_dataset

import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

dense_feature_dim = 13

args = parse_args()

role = role_maker.UserDefinedRoleMaker(
    current_id=int(os.getenv("CURRENT_ID")),
    role=role_maker.Role.WORKER
    if os.getenv("TRAINING_ROLE") == "TRAINER" else role_maker.Role.SERVER,
    worker_num=int(os.getenv("TRAINER_NUM")),
    server_endpoints=os.getenv("ENDPOINTS").split(","))

exe = fluid.Executor(fluid.CPUPlace())
fleet.init(role)

dense_input = fluid.layers.data(
    name="dense_input", shape=[dense_feature_dim], dtype='float32')
sparse_input_ids = [
    fluid.layers.data(name="C" + str(i), shape=[1], lod_level=1, dtype="int64")
    for i in range(1, 27)]
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

loss, auc_var, batch_auc_var = ctr_dnn_model_dataset(dense_input, sparse_input_ids, label,
                                                     args.embedding_size, args.sparse_feature_dim)

strategy = DistributeTranspilerConfig()
strategy.sync_mode = False

optimizer = fluid.optimizer.Adagrad(learning_rate=1e-2)
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(loss)

if fleet.is_server():
    logger.info("run pserver")
    fleet.init_server()
    with open("pserver.prog", "w") as fout:
        fout.write(str(fleet.main_program))

    fleet.run_server()
elif fleet.is_worker():
    logger.info("run trainer")

    fleet.init_worker()
    with open("worker.prog", "w") as fout:
        fout.write(str(fleet.main_program))

    exe.run(fleet.startup_program)

    input_vars = [dense_input] + sparse_input_ids + [label]
    input_var_names = [var.name for var in input_vars]
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(input_vars)
    pipe_command = "python criteo_reader.py %d" % args.sparse_feature_dim
    dataset.set_pipe_command(pipe_command)
    dataset.set_batch_size(100)
    thread_num = 5
    dataset.set_thread(thread_num)
    filelist = ["raw_data/part-%d" % x for x in range(len(os.listdir("raw_data")))]
    print("worker index: %d" % fleet.worker_index())
    print("worker num: %d" % fleet.worker_num())
    dataset.set_filelist(filelist[fleet.worker_index()::fleet.worker_num()])
    save_dirname = "ctr_model"
    epochs = 20
    print("begin to train from dataset")
    import time
    for i in range(epochs):
        time_start = time.time()
        exe.train_from_dataset(program=fleet.main_program,
                               dataset=dataset,
                               fetch_list=[auc_var],
                               fetch_info=["auc"],
                               print_period=10000,
                               debug=True)
        sys.stderr.write("epoch %d finished, use time=%d\n" % ((i + 1), time.time() - time_start))

        if fleet.worker_index() == 0:
            model_path = "%s/epoch%d.model" % (save_dirname, (i + 1))
            fluid.io.save_inference_model(model_path, input_var_names, [loss, auc_var], exe)
            if args.cloud_train:
                import paddlecloud.upload_utils as upload_utils
                sys_job_id = os.getenv("SYS_JOB_ID")
                output_path = os.getenv("OUTPUT_PATH")
                remote_path = output_path + "/" + sys_job_id + "/"
                upload_rst = upload_utils.upload_to_hdfs(local_file_path=model_path, remote_file_path=remote_path)
                logger.info("remote_path: {}, upload_rst: {}".format(remote_path, upload_rst))
    fleet.stop_worker()

