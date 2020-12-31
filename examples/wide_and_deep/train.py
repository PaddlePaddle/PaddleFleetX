
from __future__ import print_function

import os
import unittest
import time
import threading
import numpy

import paddle
paddle.enable_static()

import paddle.fluid as fluid
import paddle.distributed.fleet as fleet


from model import net
from reader import data_reader


fleet.init(role)
feeds, predict, avg_cost = net()

optimizer = fluid.optimizer.SGD(0.01)
strategy = paddle.distributed.fleet.DistributedStrategy()
strategy.a_sync = True

optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(avg_cost)

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()

if fleet.is_worker():
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    fleet.init_worker()

    train_reader = paddle.batch(fake_ctr_reader(), batch_size=4)
    reader.decorate_sample_list_generator(train_reader)

    for epoch_id in range(1):
        reader.start()
        try:
            while True:
                loss_val = exe.run(program=fluid.default_main_program(),
                                   fetch_list=[avg_cost.name])
                loss_val = np.mean(loss_val)
                print("TRAIN ---> pass: {} loss: {}\n".format(epoch_id,
                                                              loss_val))
        except fluid.core.EOFException:
            reader.reset()

    fleet.stop_worker()
