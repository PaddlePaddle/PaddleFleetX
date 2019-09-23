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
from __future__ import division

import argparse
import cProfile
import time
import os
import traceback

import numpy as np
import math
os.environ['FLAGS_sync_nccl_allreduce'] = '1'
import reader
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid.transpiler.details import program_to_code

from utility import add_arguments, print_arguments
import functools
from fast_imagenet import FastImageNet, lr_decay
import utils
from env import dist_env
from dist_utils import nccl2_prepare


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    add_arg('total_images',     int,   1281167,            "Number of training images.")
    add_arg('num_epochs',       int,   120,                "Maximum number of epochs to run.")
    add_arg('model_save_dir',   str,   "output",           "Directory to save models")
    add_arg('lr',               float, 1.0,                "Initial learning rate.")
    add_arg('data_dir',         str,   "dataset/",         "Directory for dataset.")
    add_arg('fp16',             bool,  False,              "Enable half precision training with fp16.")
    add_arg('scale_loss',       float, 64.0,               "Scale loss for fp16.")
    add_arg('update_method',    str,   "nccl2",            "Variable update method. Can be local or nccl2.")
    add_arg('start_test_pass',  int,   0,                  "At which pass to start test.")
    add_arg('log_period',       int,   30,                 "How often to print a log, default is 30.")
    add_arg('best_acc5',        float, 0.93,               "The best acc5, default is 93%.")
    add_arg('nccl_comm_num',    int,   1,                  "Number of NCCL communicators.")
    add_arg("use_hallreduce",   bool,  False,              "Use hierarchical allreduce or not.")
    add_arg("enable_backward_op_deps", bool, True,         "Whether to use enable_backward_op_deps.")
    add_arg("fuse",             bool,   False,             "Whether to use fusion.")
    add_arg("profile",          bool,   False,             "Whether to profile performance.")
    add_arg("start_profile_batch", int, 10,                "After which batch to start profiling.")
    add_arg("stop_profile_batch",  int, 16,                "After which batch to stop profiling.")
    # yapf: enable
    args = parser.parse_args()
    return args


def test_single(exe, test_args, test_reader, feeder, bs, test_prog):
    acc_evaluators = []
    for _ in range(len(test_args[1])):
        acc_evaluators.append(fluid.metrics.Accuracy())

    to_fetch = [v.name for v in test_args[1]]
    start_time = time.time()
    num_samples = 0
    for batch_id, data in enumerate(test_reader()):
        weight = len(data)
        #acc_results = exe.run(test_prog, fetch_list=to_fetch, feed=feeder.feed(data))
        acc_results = exe.run(fetch_list=to_fetch, feed=feeder.feed(data))
        ret_result = [np.mean(np.array(ret)) for ret in acc_results]
        print("Test batch: [%d], acc_result: [%s]" % (batch_id, ret_result))
        for i, e in enumerate(acc_evaluators):
            e.update(value=np.array(acc_results[i]), weight=weight)
        num_samples += weight
    print_train_time(start_time, time.time(), num_samples)

    return [e.eval() for e in acc_evaluators]


def build_program(args,
                  is_train,
                  main_program,
                  startup_program,
                  sz,
                  bs):

    img_shape = [3, sz, sz]
    class_num = 1000
    pyreader = None
    with fluid.program_guard(main_program, startup_program):
        with fluid.unique_name.guard():
            if is_train:
                pyreader = fluid.layers.py_reader(
                    capacity=bs,
                    shapes=([-1] + img_shape, (-1, 1)),
                    dtypes=('uint8', 'int64'),
                    name="train_reader_" + str(sz)
                    if is_train else "test_reader_" + str(sz),
                    use_double_buffer=True)
                img, label = fluid.layers.read_file(pyreader)
            else:
                # FIXME: for testing, why the shape is [3, 244, 244]
                img = fluid.layers.data(
                    name="image", shape=[3, 244, 244], dtype="uint8")
                label = fluid.layers.data(
                    name="label", shape=[1], dtype="int64")
            cast_img_type = "float16" if args.fp16 else "float32"
            cast = fluid.layers.cast(img, cast_img_type)
            img_mean = fluid.layers.create_global_var(
                [3, 1, 1], 0.0, cast_img_type, name="img_mean", persistable=True)
            img_std = fluid.layers.create_global_var(
                [3, 1, 1], 0.0, cast_img_type, name="img_std", persistable=True)
            t1 = fluid.layers.elementwise_sub(cast, img_mean, axis=1)
            t2 = fluid.layers.elementwise_div(t1, img_std, axis=1)

            model = FastImageNet(is_train=is_train)
            predict = model.net(t2, class_dim=class_num)
            cost, prob = fluid.layers.softmax_with_cross_entropy(
                predict, label, return_softmax=True)
            if args.scale_loss > 1.0:
                avg_cost = fluid.layers.mean(x=cost) * args.scale_loss
            else:
                avg_cost = fluid.layers.mean(x=cost)

            batch_acc1 = fluid.layers.accuracy(input=prob, label=label, k=1)
            batch_acc5 = fluid.layers.accuracy(input=prob, label=label, k=5)

            # configure optimize
            if is_train:
                total_images = args.total_images
                num_trainers = args.dist_env["num_trainers"]
                base_lr = args.lr
                lr = base_lr * num_trainers / 8

                epochs = [(0, 7), (7, 13), (13, 22), (22, 25), (25, 28)]
                bs_epoch = [bs * num_trainers for bs in [224, 224, 96, 96, 50]]
                bs_scale = [bs * 1.0 / bs_epoch[0] for bs in bs_epoch]
                lrs = [(lr, lr * 2), (lr * 2, lr / 4),
                       (lr * bs_scale[2], lr / 10 * bs_scale[2]),
                       (lr / 10 * bs_scale[2], lr / 100 * bs_scale[2]),
                       (lr / 100 * bs_scale[4], lr / 1000 * bs_scale[4]),
                       lr / 1000 * bs_scale[4]]

                boundaries, values = lr_decay(lrs, epochs, bs_epoch,
                                              total_images)
                image_per_trainer = int(math.ceil(args.total_images / num_trainers))
                step_per_trainer = int(math.ceil(image_per_trainer / 224))
                warmup_steps = step_per_trainer * 3 # 3 warmup epochs
                lr_var = utils.lr_warmup(
                        fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
                        warmup_steps, base_lr, lr)
                optimizer = fluid.optimizer.Momentum(
                    learning_rate=lr_var,
                    #learning_rate=fluid.layers.piecewise_decay(
                    #    boundaries=boundaries, values=values),
                    momentum=0.9)
                if args.fp16:
                    params_grads = optimizer.backward(avg_cost)
                    master_params_grads = utils.create_master_params_grads(
                        params_grads, main_program, startup_program, args.scale_loss)
                    optimizer.apply_gradients(master_params_grads)
                    utils.master_param_to_train_param(master_params_grads,
                                                      params_grads, main_program)
                    # optimizer = decorate(optimizer, init_loss_scaling=args.scale_loss, use_dynamic_loss_scaling=True)
                    # optimizer.minimize(avg_cost)
                else:
                    optimizer.minimize(avg_cost)

    # print("py_reader_startup_program:")
    # program_to_code(py_reader_startup_program)
    # print("startup_program:")
    # program_to_code(startup_program)
    # print("main_program:")
    # program_to_code(main_program)
    # exit()

    return avg_cost, [batch_acc1, batch_acc5], pyreader


def refresh_program(args, sz, bs, val_bs):
    train_program = fluid.Program()
    test_program = fluid.Program()
    startup_program = fluid.Program()

    train_args = build_program(
        args,
        True,
        train_program,
        startup_program,
        sz,
        bs)

    test_args = build_program(
        args,
        False,
        test_program,
        startup_program,
        sz,
        val_bs)

    gpu_id = 0
    if os.getenv("FLAGS_selected_gpus"):
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    place = fluid.CUDAPlace(gpu_id)
    startup_exe = fluid.Executor(place)

    if args.update_method == "nccl2":
        nccl2_prepare(args, startup_program, main_program=train_program)

    startup_exe.run(startup_program)
    conv2d_w_vars = [
        var for var in startup_program.global_block().vars.values()
        if var.name.startswith('conv2d_')
    ]
    for var in conv2d_w_vars:
        shape = var.shape
        if not shape or len(shape) == 0:
            fan_out = 1
        elif len(shape) == 1:
            fan_out = shape[0]
        elif len(shape) == 2:
            fan_out = shape[1]
        else:
            fan_out = shape[0] * np.prod(shape[2:])
        std = math.sqrt(2.0 / fan_out)
        kaiming_np = np.random.normal(0, std, var.shape)
        tensor = fluid.global_scope().find_var(var.name).get_tensor()
        if args.fp16 and ".master" not in var.name:
            tensor.set(np.array(
                kaiming_np, dtype="float16").view(np.uint16),
                       place)
        else:
            tensor.set(np.array(kaiming_np, dtype="float32"), place)

    np_tensors = dict()
    np_tensors["img_mean"] = np.array(
        [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0]).astype(
            "float16" if args.fp16 else "float32").reshape((3, 1, 1))
    np_tensors["img_std"] = np.array(
        [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]).astype(
            "float16" if args.fp16 else "float32").reshape((3, 1, 1))
    for var_name, np_tensor in np_tensors.items():
        var = fluid.global_scope().find_var(var_name)
        if args.fp16:
            var.get_tensor().set(np_tensor.view(np.uint16), place)
        else:
            var.get_tensor().set(np_tensor, place)

    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = args.nccl_comm_num + 1
    strategy.num_iteration_per_drop_scope = 30
    strategy.use_experimental_executor = True
    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = False
    build_strategy.enable_backward_optimizer_op_deps = args.enable_backward_op_deps
    build_strategy.reduce_strategy = fluid.BuildStrategy(
    ).ReduceStrategy.AllReduce
    if args.fuse:
        build_strategy.fuse_all_reduce_ops = True
    
    num_trainers = args.dist_env["num_trainers"]
    trainer_id = args.dist_env["trainer_id"]
    build_strategy.num_trainers = num_trainers
    build_strategy.trainer_id = trainer_id

    avg_loss = train_args[0]
    train_exe = fluid.ParallelExecutor(
        True,
        avg_loss.name,
        main_program=train_program,
        exec_strategy=strategy,
        build_strategy=build_strategy,
        num_trainers=num_trainers,
        trainer_id=trainer_id)
    test_exe = None
    if trainer_id == 0:
        #build_strategy = fluid.BuildStrategy()
        #build_strategy.enable_inplace = False
        #build_strategy.num_trainers = 1
        #build_strategy.trainer_id = 0
        #build_strategy.reduce_strategy = fluid.BuildStrategy(
        #).ReduceStrategy.AllReduce
        test_exe = fluid.ParallelExecutor(
            True,
            main_program=test_program,
            share_vars_from=train_exe,
            #exec_strategy=strategy,
            #build_strategy=build_strategy,
            exec_strategy=None,
            build_strategy=None,
            num_trainers=1,
            trainer_id=0)
    # test_exe = startup_exe

    return train_args, test_args, test_program, train_exe, test_exe


def prepare_reader(epoch_id, train_py_reader, train_bs, val_bs, trn_dir,
                   img_dim, min_scale, rect_val, args):
    num_trainers = args.dist_env["num_trainers"]
    trainer_id = args.dist_env["trainer_id"]
    train_reader = reader.train(
        traindir="%s/%strain" % (args.data_dir, trn_dir),
        sz=img_dim,
        min_scale=min_scale,
        shuffle_seed=epoch_id + 1,
        rank_id=trainer_id,
        size=num_trainers)
    train_py_reader.decorate_paddle_reader(
        paddle.batch(
            train_reader, batch_size=train_bs))

    test_reader = reader.test(
        valdir="%s/%svalidation" % (args.data_dir, trn_dir),
        bs=val_bs,
        sz=img_dim,
        rect_val=rect_val)
    test_batched_reader = paddle.batch(
        test_reader, batch_size=val_bs)

    return test_batched_reader


def train_parallel(args):
    exe = None
    test_exe = None
    train_args = None
    test_args = None

    train_args, test_args, test_program, exe, test_exe = refresh_program(
        args, sz=224, bs=224, val_bs=96)

    over_all_start = time.time()
    total_train_time = 0.0
    for epoch_id in range(args.num_epochs):
        # refresh program
        train_start_time = time.time()
        if epoch_id == 0:
            bs = 224
            val_bs = 64
            trn_dir = "160/"
            img_dim = 128
            min_scale = 0.08
            rect_val = False
        elif epoch_id == 13:
            bs = 96
            val_bs = 64
            trn_dir = "352/"
            img_dim = 224
            min_scale = 0.087
            rect_val = False
        elif epoch_id == 25:
            bs = 50
            val_bs = 8
            trn_dir = ""
            img_dim = 288
            min_scale = 0.5
            rect_val = True
        else:
            pass

        avg_loss = train_args[0]
        num_samples = 0
        iters = 0
        start_time = time.time()
        train_py_reader = train_args[2]
        test_reader = prepare_reader(
            epoch_id,
            train_py_reader,
            bs,
            val_bs,
            trn_dir,
            img_dim=img_dim,
            min_scale=min_scale,
            rect_val=rect_val,
            args=args)
        train_py_reader.start()  # start pyreader
        batch_start_time = time.time()
        while True:
            fetch_list = [avg_loss.name]
            acc_name_list = [v.name for v in train_args[1]]
            fetch_list.extend(acc_name_list)
            fetch_list.append("learning_rate")
            if iters % args.log_period == 0:
                should_print = True
            else:
                should_print = False

            fetch_ret = []
            gpu_id = int(os.getenv("FLAGS_selected_gpus"))
            try:
                if args.profile and gpu_id == 0 and iters == args.start_profile_batch and epoch_id in [0, 13, 25]:
                    profiler.start_profiler("All")
                elif args.profile and gpu_id == 0 and iters == args.stop_profile_batch and epoch_id in [0, 13, 25]:
                    profiler.stop_profiler("total", "./profile_lilong_%d" % epoch_id)
                if should_print:
                    fetch_ret = exe.run(fetch_list)
                else:
                    exe.run([])
            except fluid.core.EOFException as eof:
                print("Finish current epoch, will reset pyreader...")
                train_py_reader.reset()
                break
            except fluid.core.EnforceNotMet as ex:
                traceback.print_exc()
                exit(1)

            num_samples += bs

            if should_print:
                fetched_data = [np.mean(np.array(d)) for d in fetch_ret]
                print(
                    "Epoch %d, batch %d, loss %s, accucacys: %s, learning_rate %s, py_reader queue_size: %d, avg batch time: %0.4f secs"
                    % (epoch_id, iters, fetched_data[0], fetched_data[1:-1],
                       fetched_data[-1], train_py_reader.queue.size(),
                       (time.time() - batch_start_time) * 1.0 /
                       args.log_period))
                batch_start_time = time.time()
            iters += 1

        print_train_time(start_time, time.time(), num_samples)
        print("Epoch: %d, Spend %.5f hours (total)\n" %
              (epoch_id,
               (time.time() - over_all_start) / 3600))
        total_train_time += time.time() - train_start_time
        print("Epoch: %d, Spend %.5f hours (training only)\n" %
              (epoch_id, total_train_time / 3600))
        
        if args.dist_env["trainer_id"] == 0:
            feed_list = [
                test_program.global_block().var(var_name)
                for var_name in ("image", "label")
            ]
            gpu_id = 0
            if os.getenv("FLAGS_selected_gpus"):
                gpu_id = int(os.getenv("FLAGS_selected_gpus"))
            test_feeder = fluid.DataFeeder(
                feed_list=feed_list, place=fluid.CUDAPlace(gpu_id))
            test_ret = test_single(test_exe, test_args, test_reader, test_feeder, val_bs, test_program)
            test_acc1, test_acc5 = [np.mean(np.array(v)) for v in test_ret]
            print("Epoch: %d, Test Accuracy: %s, Spend %.2f hours\n" %
                  (epoch_id, [test_acc1, test_acc5], (time.time() - over_all_start) / 3600))
            if np.mean(np.array(test_ret[1])) > args.best_acc5:
                print("Achieve the best top-1 acc %f, top-5 acc: %f" % (test_acc1, test_acc5))
                break

    print("total train time: ", total_train_time)
    print("total run time: ", time.time() - over_all_start)


def print_train_time(start_time, end_time, num_samples):
    time_elapsed = end_time - start_time
    examples_per_sec = num_samples / time_elapsed
    print('\nTotal examples: %d, total time: %.5f, %.5f examples/sec.\n' %
          (num_samples, time_elapsed, examples_per_sec))


def print_paddle_environments():
    print('--------- Configuration Environments -----------')
    for k in os.environ:
        if "PADDLE_" in k or "FLAGS_" in k:
            print("%s: %s" % (k, os.environ[k]))
    print('------------------------------------------------')


def main():
    args = parse_args()
    print_arguments(args)
    print_paddle_environments()
    if args.update_method != 'local':
        args.dist_env = dist_env()
    train_parallel(args)


if __name__ == "__main__":
    main()
