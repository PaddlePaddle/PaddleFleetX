from model import WideDeepModel
from reader import WideDeepDataset

import paddle
paddle.enable_static()
import numpy as np
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.utils.ps_util import DistributedInfer

def distributed_training(exe, train_model, train_data_path="./data", batch_size=10, epoch_num=1):
    train_data = WideDeepDataset(data_path=train_data_path)
    reader = train_model.loader.set_sample_generator(train_data, batch_size=batch_size, drop_last=True, places=paddle.CPUPlace())
    
    for epoch_id in range(epoch_num):
        reader.start()
        try:
            while True:
                loss_val = exe.run(program=paddle.static.default_main_program(),
                                   fetch_list=[train_model.cost.name])
                loss_val = np.mean(loss_val)
                print("TRAIN ---> pass: {} loss: {}\n".format(epoch_id, loss_val))
        except paddle.common_ops_import.core.EOFException:
            reader.reset()

def distributed_infer(exe, test_model, test_data_path="./data", batch_size=10):
    test_origin_program = paddle.static.Program()
    test_startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program=test_origin_program, startup_program=test_startup_program):
        with paddle.utils.unique_name.guard():
            test_model.net(is_train=False)

    dist_infer = DistributedInfer(main_program=test_origin_program, startup_program=test_startup_program)

    test_data = WideDeepDataset(data_path=test_data_path)
    reader = test_model.loader.set_sample_generator(test_data, batch_size=batch_size, drop_last=True, places=paddle.CPUPlace())

    with paddle.static.program_guard(main_program=dist_infer.get_dist_infer_program()):
        reader.start()
        try:
            while True:
                loss_val = exe.run(program=paddle.static.default_main_program(),
                                    fetch_list=[test_model.cost.name])
                loss_val = np.mean(loss_val)
                print("TEST ---> loss: {}\n".format(loss_val))
        except paddle.common_ops_import.core.EOFException:
            reader.reset()


fleet.init(is_collective=False)

model = WideDeepModel()
model.net(is_train=True)

optimizer = paddle.optimizer.SGD(learning_rate=0.0001)

strategy = fleet.DistributedStrategy()
strategy.a_sync = True
optimizer = fleet.distributed_optimizer(optimizer, strategy)

optimizer.minimize(model.cost)


if fleet.is_server():
    fleet.init_server()
    fleet.run_server()

if fleet.is_worker():
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    exe.run(paddle.static.default_startup_program())

    fleet.init_worker()

    distributed_training(exe, model)
    distributed_infer(exe, model)

    fleet.stop_worker()
