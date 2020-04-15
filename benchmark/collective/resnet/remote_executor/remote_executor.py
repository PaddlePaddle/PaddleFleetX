from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import sys
import functools
import math
import json
import argparse
import functools
import subprocess
import paddle
import paddle.fluid as fluid

import utils.reader_cv2 as reader
import utils.utility as utility
from utils.utility import add_arguments, print_arguments, check_gpu
from utils.learning_rate import cosine_decay_with_warmup, lr_warmup
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid import compiler
from paddle.fluid.transpiler.details import program_to_code

trainer_id = int(os.environ.get('PADDLE_TRAINER_ID'))
num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('batch_size',       int,   32,                   "Minibatch size per device.")
add_arg('num_epochs',       int,   120,                  "number of epochs.")
add_arg('class_dim',        int,   1000,                 "Class number.")
add_arg('image_shape',      str,   "3,224,224",          "input image size")
add_arg('data_dir',         str,   "./data/ILSVRC2012/",  "The ImageNet dataset root dir.")
add_arg('data_format',      str,   "NCHW",               "Tensor data format when training.")
add_arg('lower_scale',      float,     0.08,      "Set the lower_scale in ramdom_crop")
add_arg('lower_ratio',      float,     3./4.,      "Set the lower_ratio in ramdom_crop")
add_arg('upper_ratio',      float,     4./3.,      "Set the upper_ratio in ramdom_crop")
add_arg('resize_short_size',      int,     256,      "Set the resize_short_size")
add_arg('mixup_alpha',      float,     0.2,      "Set the mixup_alpha parameter")
add_arg('fetch_steps',      int,  10,                "Enable profiler or not." )
add_arg('do_test',          bool,  False,                 "Whether do test every epoch.")
add_arg('image_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406], help="The mean of input image data")
add_arg('image_std', nargs='+', type=float, default=[0.229, 0.224, 0.225], help="The std of input image data")
#add_arg('interpolation',    int,  None,                 "The interpolation mode")
# These are compile arguments.
#add_arg('fuse', bool, False,                      "Whether to use tensor fusion.")
#add_arg('fuse_elewise_add_act_ops', bool, True,                      "Whether to use elementwise_act fusion.")
#add_arg('fuse_bn_act_ops', bool, True,                      "Whether to use bn_act fusion.")
#add_arg('nccl_comm_num',        int,  1,                  "nccl comm num")
#add_arg("use_hierarchical_allreduce",     bool,   False,   "Use hierarchical allreduce or not.")
#add_arg('num_threads',        int,  1,                   "Use num_threads to run the fluid program.")
#add_arg('num_iteration_per_drop_scope', int,    100,      "Ihe iteration intervals to clean up temporary variables.")
add_arg('use_mixup',      bool,      False,        "Whether to use mixup or not")
add_arg('use_dali',      bool,      False,        "Whether to use mixup or not")
add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")

def program_creator(startup_file, main_file):
    startup_prog = fluid.Program()
    main_prog = fluid.Program()
    with fluid.program_guard(main_prog, startup_prog):
        with open(startup_file, 'r') as f:
            startup_string = f.read()
            start_prog= startup_prog.parse_from_string(startup_string)

        with open(main_file, 'r') as f:
            main_string = f.read()
            main_prog = main_prog.parse_from_string(main_string)

    return start_prog, main_prog


def reader_creator(train_batch_size, places, is_train=True, data_layout="NCHW"):
    train_data_loader, data = utility.create_data_loader(is_train, args, data_layout=data_layout)
    shuffle_seed = 1 if num_trainers > 1 else None
    train_reader = reader.train(settings=args, data_dir=args.data_dir,
                                pass_id_as_seed=shuffle_seed, data_layout=args.data_format, threads=10)
    train_batch_reader=paddle.batch(train_reader, batch_size=train_batch_size)


    train_data_loader.set_sample_list_generator(train_batch_reader, places)
    return train_data_loader, None

def print_paddle_environments():
    print('--------- Configuration Environments -----------')
    #print("Devices per node: %d" % DEVICE_NUM)
    for k in os.environ:
        if "PADDLE_" in k or "FLAGS_" in k:
            print("%s: %s" % (k, os.environ[k]))
    print('------------------------------------------------')

def train(startup_file, main_file):
    # program
    startup_prog, train_prog = program_creator(startup_file, main_file)
    print("startup_program:")
    program_to_code(startup_prog)
    print("train_program:")
    program_to_code(train_prog)

    # reader
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()
    places = place
    if num_trainers <= 1 and args.use_gpu:
        places = fluid.framework.cuda_places()

    train_data_loader, _ = reader_creator(args.batch_size, places=places, data_layout=args.data_format)

    # executor
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    train_exe = exe

    train_fetch_list = ['mean_0.tmp_0', 'accuracy_0.tmp_0', 'accuracy_1.tmp_0', 'learning_rate_warmup']
    for pass_id in range(0, args.num_epochs):
        train_info = [[], [], []]
        test_info = [[], [], []]
        train_begin=time.time()
        batch_id = 0
        time_record=[]

        for data in train_data_loader():
            t1 = time.time()

            if batch_id % args.fetch_steps != 0:
                train_exe.run(train_prog, feed=data)
            else:
                loss, acc1, acc5, lr = train_exe.run(train_prog,  feed=data,  fetch_list=train_fetch_list)
                acc1 = np.mean(np.array(acc1))
                acc5 = np.mean(np.array(acc5))
                train_info[1].append(acc1)
                train_info[2].append(acc5)

            t2 = time.time()
            period = t2 - t1
            time_record.append(period)

            if batch_id % args.fetch_steps == 0:
                loss = np.mean(np.array(loss))
                train_info[0].append(loss)
                lr = np.mean(np.array(lr))
                period = np.mean(time_record)
                speed = args.batch_size * 1.0 / period
                time_record=[]
                print("Pass {0}, trainbatch {1}, loss {2}, \
                    acc1 {3}, acc5 {4}, lr {5}, time {6}, speed {7}"
                      .format(pass_id, batch_id, "%.5f"%loss, "%.5f"%acc1, "%.5f"%acc5, "%.5f" %
                                  lr, "%2.4f sec" % period, "%.2f" % speed))
                sys.stdout.flush()
            batch_id += 1

        train_loss = np.array(train_info[0]).mean()
        train_end=time.time()
        train_speed = (batch_id * train_batch_size) / (train_end - train_begin)
        train_speed_list.append(train_speed)

if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    print_paddle_environments()
    check_gpu(True)

    startup_file="../trainer_{}_startup_program.desc".format(trainer_id)
    main_file="../trainer_{}_main_program.desc".format(trainer_id)
    train(startup_file, main_file)


