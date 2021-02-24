from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from paddle.fluid.incubate.fleet.base.role_maker import GeneralRoleMaker

import numpy as np
from model import WideDeepModel
from reader import WideDeepDatasetReader
import os
import sys

import config_fleet
import paddle
paddle.enable_static()

def distributed_training(psgpu, exe, train_model, train_data_path="./data", batch_size=64, epoch_num=10, thread_num = 8):
    dataset = paddle.distributed.InMemoryDataset()
    dataset.init(use_var=train_model.inputs, pipe_command="python reader.py", batch_size=batch_size, thread_num=thread_num)
    train_files_list = [os.path.join(train_data_path, x)
                          for x in os.listdir(train_data_path)]
    dataset.set_filelist(train_files_list)
    dataset.load_into_memory()

    psgpu.set_dataset(dataset.dataset)
    psgpu.build_gpu_ps(0, 8)
    print("start train from dataset")
    for epoch_id in range(epoch_num):
        exe.train_from_dataset(paddle.static.default_main_program(),
                               dataset,
                               paddle.static.global_scope(),
                               debug=False,
                               fetch_list=[train_model.cost],
                               fetch_info=["loss"],
                               print_period=1)
    dataset.release_memory()


role_maker = GeneralRoleMaker(http_ip_port="127.0.0.1:8900")
fleet.init(role_maker)
fleet._set_client_communication_config(500000, 10000, 3)

model = WideDeepModel()
model.net(is_train=True)

optimizer = paddle.fluid.optimizer.Adam(learning_rate=5e-06, beta1=0.99, beta2=0.9999)

optimizer = fleet.distributed_optimizer(optimizer, strategy=config_fleet.config)

optimizer.minimize(model.cost, startup_programs=[paddle.static.default_startup_program()])

opt_info = paddle.static.default_main_program()._fleet_opt
opt_info["fleet_desc"].server_param.downpour_server_param.service_param.server_class = "DownpourLocalPsServer"
opt_info["fleet_desc"].server_param.downpour_server_param.service_param.client_class = "DownpourLocalPsClient"

if fleet.is_server():
    print("server run_server..")
    fleet.run_server()

if fleet.is_worker():
    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)

    exe.run(paddle.static.default_startup_program())

    fleet.init_worker()
    psgpu = paddle.fluid.core.PSGPU()
    psgpu.set_slot_vector(model.slots_name)
    psgpu.init_gpu_ps(config_fleet.config["worker_places"])

    distributed_training(psgpu, exe, model)

    fleet.stop_worker()
    print("train finished..")
