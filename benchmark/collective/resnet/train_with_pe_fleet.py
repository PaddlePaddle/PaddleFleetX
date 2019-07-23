# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import time
import os
import traceback
import functools

import numpy as np

import paddle
import paddle.fluid as fluid

from models.resnet import ResNet50
from learning_rate import lr_warmup
from reader import train, val
from utility import add_arguments, print_arguments
import paddle.fluid.compiler as compiler

from paddle.fluid.incubate.fleet.collective import fleet
from paddle.fluid.incubate.fleet.base.role_maker import UserDefinedCollectiveRoleMaker


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    add_arg('batch_size',       int,   32,                   "Mini-batch size for training.")
    add_arg('eval_batch_size',  int,   8,                    "Mini-batch size for validation")
    add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
    add_arg('total_images',     int,   1281167,              "Number of images for training.")
    add_arg('num_epochs',       int,   120,                  "Number of epochs to run.")
    add_arg('class_dim',        int,   1000,                 "Number of classes of data set.")
    add_arg('image_shape',      str,   "3,224,224",          "Size of input images.")
    add_arg('model_save_dir',   str,   "output",             "Directory to save model.")
    add_arg('lr',               float, 0.1,                  "Initial learning rate.")
    add_arg('data_dir',         str,   "./ImageNet",         "Root directory for ImageNet data set.")
    add_arg('start_test_pass',  int,   0,                    "After how many passes to start test.")
    # for distributed
    add_arg('num_threads',        int,  4,                   "Number of threads used to run fluid program.")
    add_arg('reduce_strategy',    str,  "allreduce",         "One of reduce or allreduce.")
    add_arg('enable_sequential_execution', bool, False,      "Whether to enable sequential execution.")
    # yapf: enable
    args = parser.parse_args()
    return args


def prepare_reader(is_train, pyreader, args, pass_id=1, 
                   num_trainers=1, trainer_id=0):
    if is_train:
        reader = train(
            data_dir=args.data_dir, pass_id_as_seed=pass_id, infinite=False,
            num_trainers=num_trainers, trainer_id=trainer_id)
    else:
        reader = val(data_dir=args.data_dir, parallel_test=True)
    if is_train:
        bs = args.batch_size
    else:
        bs = args.eval_batch_size
    pyreader.decorate_paddle_reader(paddle.batch(reader, batch_size=bs))


def build_program(is_train, main_prog, startup_prog, args):
    pyreader = None
    class_dim = args.class_dim
    image_shape = [int(m) for m in args.image_shape.split(",")]
    trainer_count = args.role_maker.worker_num()

    with fluid.program_guard(main_prog, startup_prog):
        pyreader = fluid.layers.py_reader(
            capacity=args.batch_size,
            shapes=([-1] + image_shape, (-1, 1)),
            dtypes=('float32', 'int64'),
            name="train_reader" if is_train else "test_reader",
            use_double_buffer=True)

        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(pyreader)
            model_def = ResNet50()
            predict = model_def.net(image, class_dim=class_dim)
            cost, pred = fluid.layers.softmax_with_cross_entropy(
                predict, label, return_softmax=True)
            avg_cost = fluid.layers.mean(x=cost)

            batch_acc1 = fluid.layers.accuracy(input=pred, label=label, k=1)
            batch_acc5 = fluid.layers.accuracy(input=pred, label=label, k=5)

            optimizer = None
            if is_train:
                global_batch_size = args.batch_size * trainer_count
                steps_per_pass = int(math.ceil(args.total_images * 1.0 / global_batch_size))
                warmup_steps = warmup_steps * 5 # warmup 5 passes
                epochs = [30, 60, 80, 90]
                bd = [steps_per_pass * e for e in epochs]

                # https://github.com/tensorflow/models/blob/4909765543ff0c96627161ecc75eec6c309dbdce/official/resnet/resnet_run_loop.py#L244
                # https://github.com/tensorflow/models/blob/4909765543ff0c96627161ecc75eec6c309dbdce/official/resnet/imagenet_main.py#L328
                batch_denom = 256
                start_lr = args.lr
                base_lr = args.lr * global_batch_size / batch_denom
                lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]

                optimizer = fluid.optimizer.Momentum(
                    learning_rate=lr_warmup(
                        fluid.layers.piecewise_decay(
                            boundaries=bd, values=lr),
                        warmup_steps,
                        start_lr,
                        base_lr),
                    momentum=0.9)
            else:
                reduced_acc1 = fluid.layers.create_tensor("float32",
                                                          name="global_acc1")
                reduced_acc5 = fluid.layers.create_tensor("float32",
                                                          name="global_acc5")
                reduced_cost = fluid.layers.create_tensor('float32', 
                                                          name='global_cost')
                fluid.layers.collective._allreduce(batch_acc1,
                                                   reduced_acc1,
                                                   "sum")
                fluid.layers.collective._allreduce(batch_acc5,
                                                   reduced_acc5,
                                                   "sum")
                fluid.layers.collective._allreduce(avg_cost,
                                                   reduced_cost,
                                                   "sum")
                batch_acc1 = fluid.layers.scale(reduced_acc1, 
                                                scale=1. / trainer_count)
                batch_acc5 = fluid.layers.scale(reduced_acc5, 
                                                scale=1. / trainer_count)
                avg_cost = fluid.layers.scale(reduced_cost, 
                                              scale=1. / trainer_count)

    # prepare reader for current program
    trainer_id = args.role_maker.worker_index()
    prepare_reader(is_train, pyreader, args, num_trainers=trainer_count, 
                   trainer_id=trainer_id)

    return pyreader, avg_cost, batch_acc1, batch_acc5, optimizer


def test_parallel(exe, test_prog, args, pyreader, fetch_list):
    acc1 = fluid.metrics.Accuracy()
    acc5 = fluid.metrics.Accuracy()
    test_losses = []
    pyreader.start()
    weight=args.eval_batch_size * args.role_maker.worker_num()
    while True:
        try:
            acc_rets = exe.run(fetch_list=fetch_list)
            test_losses.append(acc_rets[0])
            acc1.update(value=np.array(acc_rets[1]), weight=weight)
            acc5.update(value=np.array(acc_rets[2]), weight=weight)
        except fluid.core.EOFException:
            pyreader.reset()
            break
    test_avg_loss = np.mean(np.array(test_losses))
    return test_avg_loss, np.mean(acc1.eval()), np.mean(acc5.eval())


def train_parallel(args):
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    train_pyreader, train_cost, train_acc1, train_acc5, optimizer = build_program(
        True, train_prog, startup_prog, args)
    test_pyreader, test_cost, test_acc1, test_acc5, _ = build_program(
        False, test_prog, startup_prog, args)

    # For Distributed Training.
    trainer_id = args.role_maker.worker_index()
    num_trainers = args.role_maker.worker_num()
    trainer_endpoints = args.role_maker.worker_endpoints()
    fleet.init(args.role_maker)
    optimizer = fleet.distributed_optimizer(optimizer)
    optimizer.minimize(train_cost, startup_prog)

    if args.use_gpu:
        # NOTE: for multi process mode: one process per GPU device.        
        gpu_id = 0
        if os.getenv("FLAGS_selected_gpus"):
            gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()

    startup_exe = fluid.Executor(place)
    startup_exe.run(startup_prog)

    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = args.num_threads
    # num_iteration_per_drop_scope indicates how
    # many iterations to clean up the temp variables which
    # is generated during execution. It may make the execution faster,
    # because the temp variable's shape are the same between two iterations.
    strategy.num_iteration_per_drop_scope = 30

    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_sequential_execution = bool(
        args.enable_sequential_execution)

    if args.reduce_strategy == "reduce":
        build_strategy.reduce_strategy = fluid.BuildStrategy(
        ).ReduceStrategy.Reduce
    else:
        build_strategy.reduce_strategy = fluid.BuildStrategy(
        ).ReduceStrategy.AllReduce

    build_strategy.num_trainers = len(fleet.worker_endpoints())
    build_strategy.trainer_id = fleet.worker_index()
    train_prog = compiler.CompiledProgram(train_prog)
    train_prog.with_data_parallel(
        loss_name=train_cost.name,
        build_strategy=build_strategy,
        exec_strategy=strategy)

    over_all_start = time.time()
    fetch_list = [train_cost.name, train_acc1.name, train_acc5.name]

    for pass_id in range(args.num_epochs):
        num_samples = 0
        start_time = time.time()
        batch_id = 1
        train_pyreader.start()
        while True:
            try:
                if batch_id % 30 == 0:
                    fetch_ret = startup_exe.run(program=train_prog, fetch_list=fetch_list)
                    fetched_data = [np.mean(np.array(d)) for d in fetch_ret]
                    print(
                        "Pass [%d/%d], batch [%d], loss %s, acc1: %s, acc5: %s, avg batch time %.4f"
                        % (pass_id, args.num_epochs, batch_id,
                           fetched_data[0], fetched_data[1], fetched_data[2],
                           (time.time() - start_time) / batch_id))
                else:
                    startup_exe.run(program=train_prog, fetch_list=[])
            except fluid.core.EOFException:
                train_pyreader.reset()
                break
            except fluid.core.EnforceNotMet:
                traceback.print_exc()
                break
            num_samples += args.batch_size
            batch_id += 1

        print_train_time(start_time, time.time(), num_samples)
        if pass_id >= args.start_test_pass:
            test_fetch_list = [test_cost.name, test_acc1.name, test_acc5.name]
            test_ret = test_parallel(test_exe, args, 
                                     test_pyreader, test_fetch_list)
            print("Pass: %d, Test Loss %s, test acc1: %s, test acc5: %s\n" %
                  (pass_id, test_ret[0], test_ret[1], test_ret[2]))
    startup_exe.close()
    print("total train time: ", time.time() - over_all_start)


def print_train_time(start_time, end_time, num_samples):
    train_elapsed = end_time - start_time
    examples_per_sec = num_samples / train_elapsed
    print('\nTotal examples: %d, total time: %.5f, %.5f examples/sed\n' %
          (num_samples, train_elapsed, examples_per_sec))


def print_paddle_envs():
    print('----------- Configuration envs -----------')
    for k in os.environ:
        if "PADDLE_" in k:
            print("ENV %s:%s" % (k, os.environ[k]))
    print('------------------------------------------------')


def main():
    args = parse_args()
    print_arguments(args)
    print_paddle_envs()
    # Initialize role_maker
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
    trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    trainer_endpoints = trainer_endpoints.split(',')
    role_maker = UserDefinedCollectiveRoleMaker(current_id=trainer_id, 
                                                worker_endpoints=trainer_endpoints)
    args.role_maker = role_maker
    train_parallel(args)


if __name__ == "__main__":
    main()
