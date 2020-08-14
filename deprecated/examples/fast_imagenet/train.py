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
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import argparse
import time
import os
import traceback

import numpy as np
import math
import reader
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid.optimizer import Optimizer

from utility import add_arguments, print_arguments
import functools
from fast_imagenet import FastImageNet, lr_decay
from env import dist_env

from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.contrib.mixed_precision.decorator import decorate
from paddle.fluid.transpiler.details import program_to_code


parser = argparse.ArgumentParser(description="Fast ImageNet.")
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('total_images',     int,   1281167,     "Number of training images.")
add_arg('num_epochs',       int,   30,          "Maximum number of epochs to run.")
add_arg('class_dim',        int,   1000,        "Number of classes.")
add_arg('val_images',       int,   50000,       "Number of images for validation.")
add_arg('image_shape',      str,   "3,224,224", "Input image size in the format of NCHW.")
add_arg('model_save_dir',   str,   "output",    "Directory to save models.")
add_arg('data_dir',         str,   "dataset/",  "Root directory for dataset.")
add_arg('with_inplace',     bool,  False,       "Whether to use inplace memory optimization.")
add_arg('pretrained_model', str,   None,        "Directory for pretrained model.")
add_arg('start_test_pass',  int,   0,           "After which pass to start test.")
add_arg('log_period',       int,   30,          "How often to print a log.")
add_arg('best_acc5',        float, 0.93,        "The best acc5 used to early-stop the training.")
add_arg('use_fp16',         bool,  False,       "Whether to use mixed precision training.")
add_arg('scale_loss',       float, 1.0,         "Initial loss scaling for fp16.")
add_arg('use_dynamic_loss_scaling', bool, True, "Use dynamic loss scaling for fp16 or not.")
add_arg('nccl_comm_num',    int,   1,           "Number of NCCL communicators.")
add_arg('num_threads',      int,   1,           "Number of threads to run paddlepaddle.")
add_arg('data_layout',      str,   "NCHW",      "Data layout, 'NCHW' or 'NHWC'.")
add_arg('use_dali',         bool,  False,       "Whether to use Nvidia DALI.")
add_arg('lower_scale',      float, 0.08,        "Set the lower_scale in random_crop.")
add_arg('profile',          bool,  False,       "Enable profiling or not." )
add_arg('fuse',             bool, False,        "Whether to use tensor fusion.")
add_arg('fuse_elewise_add_act_ops', bool, True, "Whether to use elementwise_act fusion.")
add_arg('fuse_bn_act_ops',  bool, True,         "Whether to use bn_act fusion.")
add_arg("use_hierarchical_allreduce", bool, False,   "Use hierarchical allreduce or not.")
add_arg('num_iteration_per_drop_scope', int, 100, "The iteration intervals to clean up temporary variables.")

# yapf: enable
args = parser.parse_args()


def test_single(exe, test_args, data_iter, test_prog):
    acc_evaluators = []
    for _ in range(len(test_args[1])):
        acc_evaluators.append(fluid.metrics.Accuracy())

    to_fetch = [v.name for v in test_args[1]]
    start_time = time.time()
    num_samples = 0
    for batch_id, data in enumerate(data_iter):
        weight = len(data)
        acc_results = exe.run(test_prog, fetch_list=to_fetch,
            feed=data, use_program_cache=True)
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
                  bs,
                  data_layout="NCHW",
                  dist_strategy=None):
    img_shape = [sz, sz, 3] if data_layout == "NHWC" else [3, sz, sz]
    test_data_shape = [244, 244, 3] if data_layout == "NHWC" else [3, 244, 244]
    with fluid.program_guard(main_program, startup_program):
        with fluid.unique_name.guard():
            if is_train:
                img = fluid.layers.data(
                    name="train_image", shape=img_shape, dtype="float32")
            else:
                img = fluid.layers.data(
                    name="test_image", shape=test_data_shape, dtype="float32")
            label = fluid.layers.data(
                name="feed_label", shape=[1], dtype="int64")
            data_loader = None
            if not args.use_dali or not is_train:
                data_loader = fluid.io.DataLoader.from_generator(
                    feed_list=[img, label],
                    capacity=64,
                    use_double_buffer=True,
                    iterable=True)

            model = FastImageNet(is_train=is_train)
            predict = model.net(img, class_dim=args.class_dim,
                data_format=data_layout)
            cost, prob = fluid.layers.softmax_with_cross_entropy(
                predict, label, return_softmax=True)
            avg_cost = fluid.layers.mean(x=cost)

            batch_acc1 = fluid.layers.accuracy(input=prob, label=label, k=1)
            batch_acc5 = fluid.layers.accuracy(input=prob, label=label, k=5)

            if is_train:
                total_images = args.total_images
                num_nodes = args.num_nodes
                num_trainers = args.num_trainers
                print("lr: {}".format(args.lr))

                epochs = [(0, 7), (7, 13), (13, 22), (22, 25), (25, 28)]
                if num_nodes == 1 or num_nodes == 2:
                    bs_epoch = [bs * num_trainers 
                                for bs in [224, 224, 96, 96, 50]]
                elif num_nodes == 4:
                    bs_epoch = [int(bs * num_trainers * 0.8) 
                                for bs in [224, 224, 96, 96, 50]]
                elif num_nodes == 8:
                    bs_epoch = [int(bs * num_trainers * 0.8) 
                                for bs in [112, 112, 48, 48, 25]]
                bs_scale = [bs / bs_epoch[0] for bs in bs_epoch]
                lr = args.lr
                lrs = [(lr, lr * 2), (lr * 2, lr / 4),
                       (lr * bs_scale[2], lr / 10 * bs_scale[2]),
                       (lr / 10 * bs_scale[2], lr / 100 * bs_scale[2]),
                       (lr / 100 * bs_scale[4], lr / 1000 * bs_scale[4]),
                       lr / 1000 * bs_scale[4]]

                boundaries, values = lr_decay(lrs, epochs, bs_epoch,
                                              total_images)
                optimizer = fluid.optimizer.Momentum(
                    learning_rate=fluid.layers.piecewise_decay(
                        boundaries=boundaries, values=values),
                    momentum=0.9)
                if args.use_fp16:
                    optimizer = decorate(optimizer, 
                                         init_loss_scaling=args.scale_loss,
                                         use_dynamic_loss_scaling=args.use_dynamic_loss_scaling)
                dist_optimizer = fleet.distributed_optimizer(optimizer,
                    strategy=dist_strategy)
                dist_optimizer.minimize(avg_cost)

    return avg_cost, [batch_acc1, batch_acc5], data_loader


def refresh_program(args, sz, bs, val_bs):
    train_program = fluid.Program()
    test_program = fluid.Program()
    startup_program = fluid.Program()

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = args.num_threads
    exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

    dist_strategy = DistributedStrategy()
    dist_strategy.exec_strategy = exec_strategy
    dist_strategy.enable_inplace = args.with_inplace

    if not args.fuse:
        dist_strategy.fuse_all_reduce_ops = False
    dist_strategy.nccl_comm_num = args.nccl_comm_num
    dist_strategy.fuse_elewise_add_act_ops=args.fuse_elewise_add_act_ops
    dist_strategy.fuse_bn_act_ops =args.fuse_bn_act_ops

    role = role_maker.UserDefinedCollectiveRoleMaker(
        current_id=args.dist_env["trainer_id"],
        worker_endpoints=args.dist_env["trainer_endpoints"])
    fleet.init(role)
    train_args = build_program(
        args,
        True,
        train_program,
        startup_program,
        sz,
        bs,
        data_layout=args.data_layout,
        dist_strategy=dist_strategy)
    train_program = fleet.main_program
    with open("main.program", "w") as f:
        program_to_code(fleet._origin_program, fout=f)
    
    test_args = build_program(
        args,
        False,
        test_program,
        startup_program,
        sz,
        val_bs,
        data_layout=args.data_layout)

    gpu_id = 0
    if os.getenv("FLAGS_selected_gpus"):
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    place = fluid.CUDAPlace(gpu_id)
    startup_exe = fluid.Executor(place)

    startup_exe.run(startup_program)
    conv2d_w_vars = [
        var for var in startup_program.global_block().vars.values()
        if var.name.startswith('conv2d_') and ".cast_fp16" not in var.name
    ]
    for var in conv2d_w_vars:
        shape = var.shape
        assert len(shape) == 4
        fan_out = shape[0] * np.prod(shape[2:])
        std = math.sqrt(2.0 / fan_out)
        kaiming_np = np.random.normal(0, std, var.shape)
        tensor = fluid.global_scope().find_var(var.name).get_tensor()
        tensor.set(np.array(kaiming_np, dtype="float32"), place)

    return train_args, test_args, test_program, startup_exe, train_program, place


def prepare_reader(epoch_id, train_data_loader, test_data_loader, train_bs, val_bs, trn_dir,
                   img_dim, min_scale, rect_val, args, place, val_dir):
    num_trainers = fleet.worker_num()
    trainer_id = fleet.worker_index()
    places = place
    if not args.use_dali:
        train_reader = reader.train(
            traindir="%s/%strain" % (args.data_dir, trn_dir),
            sz=img_dim,
            min_scale=min_scale,
            shuffle_seed=epoch_id + 1,
            rank_id=trainer_id,
            size=num_trainers,
            data_layout=args.data_layout)
        train_batch_reader = paddle.batch(
                train_reader, batch_size=train_bs)
        train_data_loader.set_sample_list_generator(train_batch_reader, places)
    
    test_reader = reader.test(
        valdir=val_dir,
        bs=val_bs,
        sz=img_dim,
        rect_val=rect_val,
        data_layout=args.data_layout)
    test_batch_reader = paddle.batch(
        test_reader, batch_size=val_bs)
    test_data_loader.set_sample_list_generator(test_batch_reader, place)()


def train_parallel(args):
    exe = None
    train_args = None
    test_args = None

    train_args, test_args, test_program, exe, train_program, place = refresh_program(
        args, sz=224, bs=224, val_bs=96)

    over_all_start = time.time()
    total_train_time = 0.0
    data_dir = args.data_dir
    for epoch_id in range(args.num_epochs):
        # refresh program
        train_start_time = time.time()
        if epoch_id == 0:
            bs = 112 if args.num_nodes == 8 else 224
            val_bs = 128
            trn_dir = os.path.join(data_dir, "160")
            val_dir = os.path.join(data_dir, "160")
            img_dim = 128
            min_scale = 0.08
            rect_val = False
        elif epoch_id == 13:
            bs = 48 if args.num_nodes == 8 else 96
            val_bs = 128
            trn_dir = os.path.join(data_dir, "352")
            val_dir = os.path.join(data_dir, "352")
            img_dim = 224
            min_scale = 0.087
            rect_val = False
        elif epoch_id == 25:
            bs = 25 if args.num_nodes == 8 else 50
            val_bs = 8
            trn_dir = os.path.join(data_dir, "")
            val_dir = os.path.join(data_dir, "")
            img_dim = 288
            min_scale = 0.5
            rect_val = True
        else:
            pass

        avg_loss = train_args[0]
        num_samples = 0
        iters = 0
        start_time = time.time()
        train_data_loader = train_args[2]
        test_data_loader = test_args[2]
        prepare_reader(
            epoch_id,
            train_data_loader,
            test_data_loader,
            bs,
            val_bs,
            trn_dir,
            img_dim=img_dim,
            min_scale=min_scale,
            rect_val=rect_val,
            val_dir=val_dir,
            args=args,
            place=place)
        if args.use_dali:
            import dali
            gpu_id = args.trainer_id % 8
            image_shape = "3,%d,%d" % (img_dim, img_dim)
            print("shape", image_shape)
            train_iter = dali.train(data_dir=trn_dir,
                                    batch_size=bs,
                                    trainer_id=args.trainer_id,
                                    trainers_num=args.num_trainers,
                                    gpu_id=gpu_id,
                                    epoch_id=epoch_id,
                                    image_shape=image_shape,
                                    lower_scale=min_scale,
                                    data_layout=args.data_layout)
        else:
            train_iter = train_data_loader()
        batch_start_time = time.time()
        for data in train_iter:
            fetch_list = [avg_loss.name]
            acc_name_list = [v.name for v in train_args[1]]
            fetch_list.extend(acc_name_list)
            if iters % args.log_period == 0:
                should_print = True
            else:
                should_print = False

            fetch_ret = []
            gpu_id = int(os.getenv("FLAGS_selected_gpus"))
            if should_print:
                fetch_ret = exe.run(train_program, feed=data, fetch_list=fetch_list)
            else:
                exe.run(train_program, feed=data, fetch_list=[])

            num_samples += bs

            if should_print:
                fetched_data = [np.mean(np.array(d)) for d in fetch_ret]
                print(
                    "Epoch %d, batch %d, loss %s, accucacys: %s, "
                    "avg batch time: %0.4f secs"
                    % (epoch_id, iters, fetched_data[0], fetched_data[1:],
                       #train_py_reader.queue.size(),
                       (time.time() - batch_start_time) * 1.0 /
                       args.log_period))
                batch_start_time = time.time()
            iters += 1

        train_iter.reset()
        del train_iter
        print_train_time(start_time, time.time(), num_samples)
        print("Epoch: %d, Spend %.5f hours (total)\n" %
              (epoch_id,
               (time.time() - over_all_start) / 3600))
        total_train_time += time.time() - train_start_time
        print("Epoch: %d, Spend %.5f hours (training only)\n" %
              (epoch_id, total_train_time / 3600))
        
        trainer_id = args.dist_env["trainer_id"]
        if epoch_id >= args.start_test_pass:
            test_iter=test_data_loader
            test_ret = test_single(exe, test_args, test_iter, test_program)
            test_acc1, test_acc5 = [np.mean(np.array(v)) for v in test_ret]
            print("Epoch: %d, Test Accuracy: %s, Spend %.2f hours\n" %
                  (epoch_id, [test_acc1, test_acc5], 
                   (time.time() - over_all_start) / 3600))
            if np.mean(np.array(test_ret[1])) > args.best_acc5:
                print("Achieve the best top-1 acc %f, top-5 acc: %f" % (
                       test_acc1, test_acc5))
                if args.model_save_dir:
                    model_save_dir = args.model_save_dir
                    if not os.path.isdir(model_save_dir):
                        os.makedirs(model_save_dir)
                        fluid.io.save_persistables(exe, model_save_dir,
                                                   fleet._origin_program)
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
    global args
    print_arguments(args)
    print_paddle_environments()
    args.dist_env = dist_env()
    num_trainers = args.dist_env['num_trainers']
    num_nodes = num_trainers // 8
    args.num_nodes = num_nodes
    args.num_trainers = num_trainers
    trainer_id = args.dist_env['trainer_id']
    args.trainer_id = trainer_id
    supported_nodes = [1, 2, 4, 8]
    assert num_nodes in supported_nodes, \
        "We only support {} nodes now.".format(supported_nodes)
    if num_nodes > 1:
        args.nccl_comm_num = 2
        args.lr = 2.0
    else:
        args.nccl_comm_num = 1
        args.lr = 1.0
    train_parallel(args)


if __name__ == "__main__":
    main()

