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
import subprocess

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
    add_arg('batch_size',       int,   32,                   "Mini-batch size.")
    add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
    add_arg('total_images',     int,   1281167,              "Number of images for training.")
    add_arg('num_epochs',       int,   120,                  "Number of epochs to run.")
    add_arg('class_dim',        int,   1000,                 "Number of classes of data set.")
    add_arg('image_shape',      str,   "3,224,224",          "Size of input images.")
    add_arg('model_save_dir',   str,   "output",             "Directory to save model.")
    add_arg('with_mem_opt',     bool,  False,                "Whether to use memory optimization or not.")
    add_arg('lr',               float, 0.1,                  "Initial learning rate.")
    add_arg('lr_strategy',      str,   "piecewise_decay",    "Learning rate decay strategy to use.")
    add_arg('model',            str,   "ResNet50",           "Network to use.")
    add_arg('enable_ce',        bool,  False,                "If set True, enable continuous evaluation job.")
    add_arg('data_dir',         str,   "./ImageNet",         "Root directory for ImageNet data set.")
    # for distributed
    add_arg('multi_batch_repeat', int,  1,                   "Batch merge repeats.")
    add_arg('num_threads',        int,  4,                   "Number of threads used to run fluid program.")
    add_arg('reduce_strategy',    str,  "allreduce",         "One of reduce or allreduce.")
    add_arg('enable_sequential_execution', bool, False,      "Skip data not if data not balanced on nodes.")
    #for dgc
    add_arg('enable_dgc', bool, False,                       "Skip data not if data not balanced on nodes.")
    add_arg('rampup_begin_step', int, 5008,                  "Skip data not if data not balanced on nodes.")
    # yapf: enable
    args = parser.parse_args()
    return args


def get_device_num():
    if os.getenv("CPU_NUM"):
        return int(os.getenv("CPU_NUM"))
    visible_device = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(
            ['nvidia-smi', '-L']).decode().count('\n')
    return device_num


def prepare_reader(is_train, pyreader, args, pass_id=1, num_trainers=1, trainer_id=0):
    if is_train:
        reader = train(
            data_dir=args.data_dir, pass_id_as_seed=pass_id, infinite=False,
            num_trainers=num_trainers, trainer_id=trainer_id)
    else:
        reader = val(data_dir=args.data_dir)
    if is_train:
        bs = args.batch_size
    else:
        bs = 8
    pyreader.decorate_paddle_reader(paddle.batch(reader, batch_size=bs))


def build_program(is_train, main_prog, startup_prog, args):
    pyreader = None
    class_dim = args.class_dim
    image_shape = [int(m) for m in args.image_shape.split(",")]

    trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))

    device_num_per_worker = get_device_num()
    with fluid.program_guard(main_prog, startup_prog):
        pyreader = fluid.layers.py_reader(
            capacity=args.batch_size if is_train else 8,
            shapes=([-1] + image_shape, (-1, 1)),
            dtypes=('float32', 'int64'),
            name="train_reader" if is_train else "test_reader",
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(pyreader)

            # FIXME: to add VGG16
            model_def = ResNet50()

            predict = model_def.net(image, class_dim=class_dim)
            cost, pred = fluid.layers.softmax_with_cross_entropy(
                predict, label, return_softmax=True)
            avg_cost = fluid.layers.mean(x=cost)

            batch_acc1 = fluid.layers.accuracy(input=pred, label=label, k=1)
            batch_acc5 = fluid.layers.accuracy(input=pred, label=label, k=5)

            optimizer = None
            if is_train:
                start_lr = args.lr
                end_lr = args.lr * trainer_count * args.multi_batch_repeat
                if os.getenv("FLAGS_selected_gpus"):
                    # in multi process mode, "trainer_count" will be total devices
                    # in the whole cluster, and we need to scale num_of nodes.
                    end_lr /= device_num_per_worker

                total_images = args.total_images / trainer_count
                if os.getenv("FLAGS_selected_gpus"):
                    step = int(total_images /
                               (args.batch_size / device_num_per_worker *
                                args.multi_batch_repeat) + 1)
                else:
                    step = int(total_images / (args.batch_size *
                                               args.multi_batch_repeat) + 1)
                warmup_steps = step * 5  # warmup 5 passes
                epochs = [30, 60, 80]
                bd = [step * e for e in epochs]
                base_lr = end_lr
                lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
                print("start lr: %s, end lr: %s, decay boundaries: %s" %
                      (start_lr, end_lr, bd))

                # NOTE: we put weight decay in layers config, and remove
                # weight decay on bn layers, so don't add weight decay in
                # optimizer config.

                optimizer = fluid.optimizer.Momentum(
                    learning_rate=lr_warmup(
                        fluid.layers.piecewise_decay(
                            boundaries=bd, values=lr),
                        warmup_steps,
                        start_lr,
                        end_lr),
                    momentum=0.9)

                if args.enable_dgc:
                    optimizer = fluid.optimizer.DGCMomentumOptimizer(
                        learning_rate=lr_warmup(
                            fluid.layers.piecewise_decay(
                                boundaries=bd, values=lr),
                            warmup_steps,
                            start_lr,
                            end_lr),
                        momentum=0.9,
                        sparsity=[0.999, 0.999],
                        rampup_begin_step=args.rampup_begin_step)

    # prepare reader for current program
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
    prepare_reader(is_train, pyreader, args, num_trainers=num_trainers, trainer_id=trainer_id)

    return pyreader, avg_cost, batch_acc1, batch_acc5, optimizer


def train_parallel(args):
    train_prog = fluid.Program()
    startup_prog = fluid.Program()

    train_pyreader, train_cost, train_acc1, train_acc5, optimizer = build_program(
        True, train_prog, startup_prog, args)

    # For Distributed Training.
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
    num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM"))
    trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    trainer_endpoints = trainer_endpoints.split(',')
    role_maker = UserDefinedCollectiveRoleMaker(current_id=trainer_id, worker_endpoints=trainer_endpoints)
    # fleet = fleet()
    fleet.init(role_maker)
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

    if args.multi_batch_repeat > 1:
        pass_builder = build_strategy._finalize_strategy_and_create_passes()
        mypass = pass_builder.insert_pass(
            len(pass_builder.all_passes()) - 4, "multi_batch_merge_pass")
        mypass.set("num_repeats", args.multi_batch_repeat)

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
        #if pass_id == 0:
        #    train_pyreader.start()
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
    train_parallel(args)


if __name__ == "__main__":
    main()
