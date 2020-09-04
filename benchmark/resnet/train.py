# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
os.environ['FLAGS_cudnn_exhaustive_search'] = "1"
os.environ['FLAGS_conv_workspace_size_limit'] = "4000"
os.environ['FLAGS_cudnn_batchnorm_spatial_persistent'] = "1"
os.environ['FLAGS_fuse_parameter_memory_size'] = "32"
os.environ['FLAGS_fuse_parameter_groups_size'] = "50"

import argparse
import six
import fleetx as X
import numpy as np
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import math
import time


def parse_args():
    parser = argparse.ArgumentParser("resnet")

    parser.add_argument(
        "--epochs", type=int, default=90, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Minibatch size per device")
    parser.add_argument(
        "--total_images", type=int, default=1281167, help="Training image number.")
    parser.add_argument(
        "--lr", type=float, default=0.1, help="base learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum value")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay")

    parser.add_argument(
        "--use_amp", type=bool, default=False, help="Whether use amp(Automatic Mixed Precision)")
    parser.add_argument(
        "--use_dgc", type=bool, default=False, help="Whether use dgc(Deep Gradient Compression)")
    parser.add_argument(
        "--dgc_rampup_epoch", type=int, default=4, help="The beginning epoch dgc implemented.")

    args = parser.parse_args()
    return args


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in six.iteritems(vars(args)):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def lr_warmup(learning_rate, warmup_steps, start_lr, end_lr):
    """ Applies linear learning rate warmup for distributed training
        Argument learning_rate can be float or a Variable
        lr = lr + (warmup_rate * step / warmup_steps)
    """
    assert (isinstance(end_lr, float))
    assert (isinstance(start_lr, float))
    linear_step = end_lr - start_lr
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate_warmup")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                decayed_lr = start_lr + linear_step * (global_step /
                                                       warmup_steps)
                fluid.layers.tensor.assign(decayed_lr, lr)
            with switch.default():
                fluid.layers.tensor.assign(learning_rate, lr)

        return lr


def lr_strategy(args, global_batch_size):
    steps_per_pass = int(math.ceil(args.total_images * 1.0 / global_batch_size))
    warmup_steps = steps_per_pass * 5

    passes = [30, 60, 80, 90]
    bd = [steps_per_pass * p for p in passes]

    batch_denom = 256
    start_lr = args.lr
    base_lr = args.lr * global_batch_size / batch_denom
    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
    lr_var = lr_warmup(fluid.layers.piecewise_decay(boundaries=bd, values=lr),
                       warmup_steps, start_lr, base_lr)
    return lr_var


def main(args):
    # init distributed
    fleet.init(is_collective=True)
    rank, nranks = fleet.worker_index(), fleet.worker_num()
    
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.amp = args.use_amp
    dist_strategy.dgc = args.use_dgc
    
    # define model
    model = X.applications.Resnet50()
    model.test_prog = model.main_prog.clone(for_test=True)
    
    # define data loader
    loader = model.load_imagenet_from_file("./ImageNet/train.txt", batch_size=args.batch_size)#, data_layout='NCHW')
    test_loader = model.load_imagenet_from_file("./ImageNet/val.txt", phase='val', batch_size=args.batch_size)#, data_layout='NCHW')
    
    lr = lr_strategy(args, args.batch_size * nranks)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr,
        momentum=args.momentum,
        regularization=fluid.regularizer.L2Decay(args.weight_decay))
    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    optimizer.minimize(model.loss)
    
    place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    exe = fluid.Executor(place)
    
    exe.run(model.startup_prog)
    print("============start training============")
    for epoch_id in range(args.epochs):
        fetch_vars = [model.loss] + model.target

        train_loss, train_acc1, train_acc5 = [], [], []
        total_time = 0
        for step, data in enumerate(loader()):
            start_time = time.time()
            loss, acc1, acc5 = exe.run(model.main_prog,
                                       feed=data,
                                       fetch_list=fetch_vars,
                                       use_program_cache=True)

            loss, acc1, acc5 = np.mean(loss), np.mean(acc1), np.mean(acc5)
            train_loss.append(loss)
            train_acc1.append(acc1)
            train_acc5.append(acc5)

            end_time = time.time()
            total_time += (end_time - start_time)

            if step % 10 == 0:
                print("Train epoch %d, step %d, loss %f, acc1 %f, acc5 %f, total time cost %f, average speed %f"
                    % (epoch_id, step, loss, acc1, acc5, total_time, (step+1) * args.batch_size / total_time))

        train_loss = np.array(train_loss).mean()
        train_acc1 = np.array(train_acc1).mean()
        train_acc5 = np.array(train_acc5).mean()
        print("End train epoch {}, loss {}, acc1 {}, acc5 {}, average speed {}".format(
                epoch_id, train_loss, train_acc1, train_acc5, (step+1) * args.batch_size / total_time))
    
        if rank == 0 and (epoch_id % 5 == 0 or epoch_id >= args.epochs - 10):
            test_loss, test_acc1, test_acc5 = [], [], []
            for j, test_data in enumerate(test_loader()):
                t1 = time.time()
                loss, acc1, acc5 = exe.run(model.test_prog,
                                           feed=test_data,
                                           fetch_list=fetch_vars,
                                           use_program_cache=True)
                loss, acc1, acc5 = np.mean(loss), np.mean(acc1), np.mean(acc5)
                test_loss.append(loss)
                test_acc1.append(acc1)
                test_acc5.append(acc5)
                period = time.time() - t1
                print("Test epoch %d, step %d, loss %f, acc1 %f, acc5 %f, test speed %f"
                    % (epoch_id, j, loss, acc1, acc5, args.batch_size / period))
            test_loss = np.array(test_loss).mean()
            test_acc1 = np.array(test_acc1).mean()
            test_acc5 = np.array(test_acc5).mean()
            print("End test epoch {}, loss {}, acc1 {}, acc5 {}".format(
                    epoch_id, test_loss, test_acc1, test_acc5))


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)

    main(args)
