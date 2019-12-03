#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import os
import time
import numpy as np
import logging
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
from network import CTR
from argument import params_args

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def get_dataset(inputs, params):
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python dataset_generator.py")
    dataset.set_batch_size(params.batch_size)
    thread_num = int(params.cpu_num)
    dataset.set_thread(thread_num)
    file_list = [
        str(params.train_files_path) + "/%s" % x
        for x in os.listdir(params.train_files_path)
    ]
    # 请确保每一个训练节点都持有不同的训练文件
    # 当我们用本地多进程模拟分布式时，每个进程需要拿到不同的文件
    # 使用 fleet.split_files 可以便捷的以文件为单位分配训练样本
    if not int(params.cloud):
        file_list = fleet.split_files(file_list)
    dataset.set_filelist(file_list)
    logger.info("file list: {}".format(file_list))
    return dataset


def train(params):
    # 根据环境变量确定当前机器/进程在分布式训练中扮演的角色
    # 然后使用 fleet api的 init()方法初始化这个节点
    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)

    # 我们还可以进一步指定分布式的运行模式，通过 DistributeTranspilerConfig进行配置
    # 如下，我们设置分布式运行模式为异步(async)，同时将参数进行切分，以分配到不同的节点
    strategy = DistributeTranspilerConfig()
    strategy.sync_mode = False
    strategy.runtime_split_send_recv = True

    ctr_model = CTR()
    inputs = ctr_model.input_data(params)
    avg_cost, auc_var, batch_auc_var = ctr_model.net(inputs, params)
    optimizer = fluid.optimizer.Adam(params.learning_rate)
    # 配置分布式的optimizer，传入我们指定的strategy，构建program
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(avg_cost)

    # 根据节点角色，分别运行不同的逻辑
    if fleet.is_server():
        # 初始化及运行参数服务器节点
        fleet.init_server()
        fleet.run_server()

    elif fleet.is_worker():
        # 初始化工作节点
        fleet.init_worker()

        exe = fluid.Executor(fluid.CPUPlace())
        # 初始化含有分布式流程的fleet.startup_program
        exe.run(fleet.startup_program)
        dataset = get_dataset(inputs, params)

        for epoch in range(params.epochs):
            start_time = time.time()

            # 训练节点运行的是经过分布式裁剪的fleet.mian_program
            exe.train_from_dataset(program=fleet.main_program,
                                   dataset=dataset,
                                   fetch_list=[auc_var],
                                   fetch_info=["Epoch {} auc ".format(epoch)],
                                   print_period=100,
                                   debug=False)
            end_time = time.time()
            logger.info("epoch %d finished, use time=%d\n" %
                        ((epoch), end_time - start_time))

            # 默认使用0号节点保存模型
            if params.test and fleet.is_first_worker():
                model_path = (str(params.model_path) + "/" + "epoch_" +
                              str(epoch))
                fluid.io.save_persistables(executor=exe, dirname=model_path)

        fleet.stop_worker()
        logger.info("Distribute Train Success!")


if __name__ == "__main__":
    params = params_args()
    train(params)
