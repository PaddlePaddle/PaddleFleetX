from paddle.distributed.fleet.utils.ps_util import DistributedInfer
import paddle.distributed.fleet as fleet
import numpy as np
from model import WideDeepModel
from reader import WideDeepDataset

import paddle
paddle.enable_static()


def distributed_training(exe, train_model, train_data_path="./data", batch_size=10, epoch_num=1):
    train_data = WideDeepDataset(data_path=train_data_path)
    reader = train_model.loader.set_sample_generator(
        train_data, batch_size=batch_size, drop_last=True, places=paddle.CPUPlace())

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
    place = paddle.CPUPlace()

    with paddle.static.program_guard(main_program=test_origin_program, startup_program=test_startup_program):
        with paddle.utils.unique_name.guard():
            test_model.net(is_train=False)

    dist_infer = DistributedInfer(
        main_program=test_origin_program, startup_program=test_startup_program)

    test_data = WideDeepDataset(data_path=test_data_path)
    reader = test_model.loader.set_sample_generator(
        test_data, batch_size=batch_size, drop_last=True, places=place)

    batch_idx = 0
    with paddle.static.program_guard(main_program=dist_infer.get_dist_infer_program()):
        reader.start()
        try:
            while True:

                loss_val, auc_val, acc_val, mae_val, mse_val, rmse_val = exe.run(program=paddle.static.default_main_program(),
                                                                                 fetch_list=[test_model.cost.name,
                                                                                             test_model._metrics["auc"]["result"].name,
                                                                                             test_model._metrics["acc"]["result"].name,
                                                                                             test_model._metrics["mae"]["result"].name,
                                                                                             test_model._metrics["mse"]["result"].name,
                                                                                             test_model._metrics["rmse"]["result"].name, ])

                print("TEST ---> loss: {} auc: {} acc: {} mae: {}, mse: {} rmse: {}\n".format(np.mean(loss_val),
                                                                                              np.mean(auc_val), np.mean(acc_val), np.mean(mae_val), np.mean(mse_val), np.mean(rmse_val)))

                batch_idx += 1
                if batch_idx % 5 == 0:
                    avg_loss = fleet.metrics.sum(
                        loss_val) / float(fleet.worker_num())
                    global_auc = fleet.metrics.auc(test_model._metrics["auc"]["state"]["stat_pos"][0],
                                                   test_model._metrics["auc"]["state"]["stat_neg"][0])
                    global_acc = fleet.metrics.acc(test_model._metrics["acc"]["state"]["correct"][0],
                                                   test_model._metrics["acc"]["state"]["total"][0])
                    global_mae = fleet.metrics.mae(test_model._metrics["mae"]["state"]["abserr"][0],
                                                   test_model._metrics["mae"]["state"]["total"][0])
                    global_mse = fleet.metrics.mse(test_model._metrics["mse"]["state"]["sqrerr"][0],
                                                   test_model._metrics["mse"]["state"]["total"][0])
                    global_rmse = fleet.metrics.rmse(test_model._metrics["rmse"]["state"]["sqrerr"][0],
                                                     test_model._metrics["rmse"]["state"]["total"][0])
                    print("Global Metrics ---> average loss: {} global auc: {} global acc: {} global mae: {} global mse: {} global rmse: {}\n".format(avg_loss,
                                                                                                                                                      global_auc, global_acc, global_mae, global_mse, global_rmse))

        except paddle.common_ops_import.core.EOFException:
            reader.reset()


def clear_metric_state(model, place):
    for metric_name in model._metrics:
        for _, state_var_tuple in model._metrics[metric_name]["state"].items():
            var = paddle.static.global_scope().find_var(
                state_var_tuple[0].name)
            if var is None:
                continue
            var = var.get_tensor()
            data_zeros = np.zeros(var._get_dims()).astype(state_var_tuple[1])
            var.set(data_zeros, place)


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
    clear_metric_state(model, place)
    distributed_infer(exe, model)

    fleet.stop_worker()
