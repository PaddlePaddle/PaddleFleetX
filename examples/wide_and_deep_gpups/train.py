import paddle
import paddle.distributed.fleet as fleet

import numpy as np
from model import WideDeepModel
from reader import WideDeepDatasetReader
import os
import sys

import paddle
paddle.enable_static()

def distributed_training(psgpu, exe, train_model, train_data_path="./data", batch_size=64, epoch_num=10, thread_num = 8):
    dataset = paddle.distributed.InMemoryDataset()
    dataset._set_use_ps_gpu(True)
    dataset.init(use_var=train_model.inputs, pipe_command="python3 reader.py", batch_size=batch_size, thread_num=thread_num)
    train_files_list = [os.path.join(train_data_path, x)
                          for x in os.listdir(train_data_path)]
    dataset.set_filelist(train_files_list)
    dataset.load_into_memory()
    
    psgpu.begin_pass()
    for epoch_id in range(epoch_num):
        exe.train_from_dataset(paddle.static.default_main_program(),
                               dataset,
                               paddle.static.global_scope(),
                               debug=False,
                               fetch_list=[train_model.cost],
                               fetch_info=["loss"],
                               print_period=1)
    psgpu.end_pass()
    dataset.release_memory()
    psgpu.finalize()


fleet.init()

model = WideDeepModel()
model.net(is_train=True)

strategy = fleet.DistributedStrategy()
strategy.a_sync = False
strategy.a_sync_configs = {"use_ps_gpu": 1}

optimizer = paddle.optimizer.SGD(learning_rate=0.0001)
optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

optimizer.minimize(model.cost)

if fleet.is_server():
    print("server run_server..")
    fleet.run_server()

if fleet.is_worker():
    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)

    exe.run(paddle.static.default_startup_program())

    fleet.init_worker()
    psgpu = paddle.fluid.core.PSGPU()

    distributed_training(psgpu, exe, model)

    fleet.stop_worker()
    print("train finished..")
