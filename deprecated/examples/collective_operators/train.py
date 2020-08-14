#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import sys
import functools
import math

def set_paddle_flags(flags):
    for key, value in flags.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect. 
set_paddle_flags({
    'FLAGS_eager_delete_tensor_gb': 0,  # enable gc 
    'FLAGS_fraction_of_gpu_memory_to_use': 0.98
})
import argparse
import subprocess
import paddle
import paddle.fluid as fluid
import models
import utils.reader_cv2 as reader
from utils.utility import add_arguments, print_arguments, check_gpu
from utils.learning_rate import lr_warmup
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker

num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
trainer_id = int(os.environ.get('PADDLE_TRAINER_ID'))

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('batch_size',                   int,    32,                   "Minibatch size per device.")
add_arg('total_images',                 int,    1281167,              "Training image number.")
add_arg('num_epochs',                   int,    120,                  "number of epochs.")
add_arg('class_dim',                    int,    1000,                 "Class number.")
add_arg('image_shape',                  str,    "3,224,224",          "input image size")
add_arg('model_save_dir',               str,    "output",             "model save directory")
add_arg('pretrained_model',             str,    "./output/ResNet50/1",                 "Whether to use pretrained model.")
add_arg('checkpoint',                   str,    None,                 "Whether to resume checkpoint.")
add_arg('lr',                           float,  0.1,                  "set learning rate.")
add_arg('model',                        str,    "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('data_dir',                     str,    "./data/ILSVRC2012/", "The ImageNet dataset root dir.")
add_arg('l2_decay',                     float,  1e-4,                 "L2_decay parameter.")
add_arg('momentum_rate',                float,  0.9,                  "momentum_rate.")
add_arg('lower_scale',                  float,  0.08,                 "Set the lower_scale in ramdom_crop")
add_arg('lower_ratio',                  float,  3./4.,                "Set the lower_ratio in ramdom_crop")
add_arg('upper_ratio',                  float,  4./3.,                "Set the upper_ratio in ramdom_crop")
add_arg('resize_short_size',            int,    256,                  "Set the resize_short_size")
add_arg('use_gpu',                      bool,   True,                 "Whether to use GPU or not.")
add_arg('nccl_comm_num',                int,    1,                    "nccl comm num")
add_arg('num_iteration_per_drop_scope', int,    30,                   "Ihe iteration intervals to clean up temporary variables.")
add_arg('is_Test',                      bool,   False,                "Whether to test on every epoch")


def optimizer_setting(params):
    l2_decay = params["l2_decay"]
    momentum_rate = params["momentum_rate"]
    # piecewise_decay
    global_batch_size = params["batch_size"] * num_trainers
    steps_per_pass = int(math.ceil(params["total_images"] * 1.0 / global_batch_size)) 
    warmup_steps = steps_per_pass * 5
    passes = [30,60,80,90]
    bd = [steps_per_pass * p for p in passes]
    batch_denom = 256
    start_lr = params["lr"]
    base_lr = params["lr"] * global_batch_size / batch_denom
    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
    lr_var = lr_warmup(fluid.layers.piecewise_decay(boundaries=bd, values=lr),
                       warmup_steps, start_lr, base_lr)
    optimizer = fluid.optimizer.Momentum(learning_rate=lr_var,momentum=momentum_rate,
                                        regularization=fluid.regularizer.L2Decay(l2_decay))
    return optimizer

def net_config(image, model, args, is_train, label=0, y_a=0, y_b=0, lam=0.0):
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not lists: {}".format(args.model,
                                                                  model_list)
    class_dim = args.class_dim
    model_name = args.model

    out = model.net(input=image, class_dim=class_dim)
    softmax_out = fluid.layers.softmax(out, use_cudnn=False)
    if is_train:
        cost, prob = fluid.layers.softmax_with_cross_entropy(out, label, return_softmax=True) 
    else:
        cost = fluid.layers.cross_entropy(input=softmax_out, label=label)

    avg_cost = fluid.layers.mean(cost)
    acc_top1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=softmax_out, label=label, k=5)
    return avg_cost, acc_top1, acc_top5

def build_program(is_train, main_prog, startup_prog, args, dist_strategy=None):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_name = args.model
    model_list = [m for m in dir(models) if "__" not in m]
    model = models.__dict__[model_name]()
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True)

        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(py_reader)
            avg_cost, acc_top1, acc_top5 = net_config(image, model, args, label=label, is_train=is_train)
            avg_cost.persistable = True
            acc_top1.persistable = True
            acc_top5.persistable = True
            build_program_out = [py_reader, avg_cost, acc_top1, acc_top5]

            if is_train:
                params = model.params
                params["total_images"] = args.total_images
                params["lr"] = args.lr
                params["num_epochs"] = args.num_epochs
                params["batch_size"] = args.batch_size
                params["l2_decay"] = args.l2_decay
                params["momentum_rate"] = args.momentum_rate
                optimizer = optimizer_setting(params)
                global_lr = optimizer._global_learning_rate()
                if num_trainers > 1:
                    dist_optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
                    _, param_grads = dist_optimizer.minimize(avg_cost)
                else:
                    print("This is only one card, all strategies are cancelled.")
                    optimizer.minimize(avg_cost)
                global_lr.persistable=True
                build_program_out.append(global_lr)

    return build_program_out

def get_device_num():
    """
    # NOTE(zcd): for multi-processe training, each process use one GPU card.
    if num_trainers > 1 : return 1
    visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(['nvidia-smi','-L']).decode().count('\n')
    """
    device_num = fluid.core.get_cuda_device_count()
    return device_num

def train(args):
    # parameters from arguments
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

    dist_strategy = DistributedStrategy()
    dist_strategy.mode = "collective"
    dist_strategy.collective_mode = "grad_allreduce"
    dist_strategy.nccl_comm_num = args.nccl_comm_num
    dist_strategy.exec_strategy = exec_strategy

    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)

    b_out = build_program(
                     is_train=True,
                     main_prog=train_prog,
                     startup_prog=startup_prog,
                     args=args,
                     dist_strategy=dist_strategy)
    train_py_reader, train_cost, train_acc1, train_acc5, global_lr = b_out[0],b_out[1],b_out[2],b_out[3],b_out[4]
    train_fetch_vars = [train_cost, train_acc1, train_acc5, global_lr]
    train_fetch_list = []

    for var in train_fetch_vars:
        var.persistable=True
        train_fetch_list.append(var.name)

    if num_trainers > 1: 
        train_prog = fleet.main_program
    else:
        pass

    b_out_test = build_program(
                     is_train=False,
                     main_prog=test_prog,
                     startup_prog=startup_prog,
                     args=args)
    test_py_reader, test_cost, test_acc1, test_acc5 = b_out_test[0],b_out_test[1],b_out_test[2],b_out_test[3]
    test_prog = test_prog.clone(for_test=True)

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if checkpoint is not None:
        fluid.io.load_persistables(exe, checkpoint, main_program=train_prog)

    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(
            exe, pretrained_model, main_program=train_prog, predicate=if_exist)

    if args.use_gpu:
        device_num = get_device_num()
    else:
        device_num = 1

    train_batch_size = args.batch_size
    print("train_batch_size: %d device_num:%d" % (train_batch_size, device_num))

    test_batch_size = 16
    # NOTE: the order of batch data generated by batch_reader
    # must be the same in the respective processes.
    shuffle_seed = 1 if num_trainers > 1 else None

    train_reader = reader.train(settings=args, data_dir=args.data_dir, pass_id_as_seed=shuffle_seed,num_epoch=args.num_epochs+1)
    test_reader = reader.val(settings=args, data_dir=args.data_dir)
    
    train_py_reader.decorate_paddle_reader(paddle.batch(train_reader,
                                                        batch_size=train_batch_size))
    test_py_reader.decorate_paddle_reader(paddle.batch(test_reader,
                                                       batch_size=test_batch_size))

    test_fetch_vars = [test_cost, test_acc1, test_acc5]
    test_fetch_list = []
    for var in test_fetch_vars:
        var.persistable=True
        test_fetch_list.append(var.name)
    train_exe = exe

    step_cnt = 0
    params = models.__dict__[args.model]().params
    global_batch_size = args.batch_size * num_trainers
    steps_per_pass = int(math.ceil(args.total_images * 1.0 / global_batch_size))
    print("steps_per_pass  {}".format(steps_per_pass))
    print("global_batch_size {}".format(global_batch_size))

    pass_id = 0
    all_train_time = []
    try :
        train_py_reader.start()
        train_info = [[], [], []]
        test_info = [[], [], []]
        train_begin=time.time()
        batch_id = 0
        time_record=[]
        while True:
            t1 = time.time()
            pass_id =  step_cnt // steps_per_pass
            if pass_id >= args.num_epochs:
                train_py_reader.reset()
                print("Train is over. Time is {}".format(all_train_time))
                break
            
            loss, acc1, acc5, lr = train_exe.run(train_prog, fetch_list=train_fetch_list, use_program_cache=True)
       
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            train_info[1].append(acc1)
            train_info[2].append(acc5)
           
            t2 = time.time() 
            period = t2 - t1
            time_record.append(period)

            loss = np.mean(np.array(loss))
            train_info[0].append(loss)
            lr = np.mean(np.array(lr))

            if batch_id % 30 == 0:
                period = np.mean(time_record)
                speed = args.batch_size * 1.0 / period
                time_record=[]
                print("Pass {0}, trainbatch {1}, loss {2}, acc1 {3}, acc5 {4}, lr {5}, time {6}, speed {7}"\
                       .format(pass_id, batch_id, "%.5f"%loss, "%.5f"%acc1, "%.5f"%acc5, "%.5f" %\
                       lr, "%2.2f sec" % period, "%.2f" % speed))
                sys.stdout.flush()
            batch_id += 1
            step_cnt += 1
            if (step_cnt // steps_per_pass) != pass_id:   # train epoch end
                train_loss = np.array(train_info[0]).mean()
                train_acc1 = np.array(train_info[1]).mean()
                train_acc5 = np.array(train_info[2]).mean()
                train_end=time.time()
                all_train_time.append(train_end - train_begin)
                train_speed = (batch_id * train_batch_size) / (train_end - train_begin)
                print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, "
                      "speed {4}".format(\
                      pass_id, "%.5f"%train_loss, "%.5f"%train_acc1, "%.5f"%train_acc5, "%.2f" % train_speed))
                sys.stdout.flush()
                #init 
                batch_id = 0
                train_info = [[], [], []]
                train_begin=time.time()
                batch_id = 0
                time_record=[]

                if args.is_Test:
                    test_info = [[], [], []]
                    test_py_reader.start()
                    test_batch_id = 0
                    try:
                        while True:
                            t1 = time.time()
                            loss, acc1, acc5 = exe.run(program=test_prog,
                                                       fetch_list=test_fetch_list,
                                                       use_program_cache=True)
                            t2 = time.time()
                            period = t2 - t1
                            loss = np.mean(loss)
                            acc1 = np.mean(acc1)
                            acc5 = np.mean(acc5)
                            test_info[0].append(loss)
                            test_info[1].append(acc1)
                            test_info[2].append(acc5)

                            if test_batch_id % 200 == 0:
                                test_speed = test_batch_size * 1.0 / period
                                print("Pass {0},testbatch {1},loss {2}, acc1 {3},acc5 {4},time {5},speed {6}"\
                                     .format(pass_id, test_batch_id, "%.5f"%loss,"%.5f"%acc1, "%.5f"%acc5,\
                                     "%2.2f sec" % period, "%.2f" % test_speed))
                                sys.stdout.flush()
                            test_batch_id += 1
                    except fluid.core.EOFException:
                        test_py_reader.reset()

                    test_loss = np.array(test_info[0]).mean()
                    test_acc1 = np.array(test_info[1]).mean()
                    test_acc5 = np.array(test_info[2]).mean()

                    print("End pass {0}, test_loss {1}, test_acc1 {2}, test_acc5 {3}".format(pass_id,"%.5f"%test_loss,
                          "%.5f"%test_acc1, "%.5f"%test_acc5))
                    sys.stdout.flush()
    except fluid.core.EOFException:
        train_py_reader.reset()
    #start test
    test_py_reader.start()
    test_batch_id = 0
    test_info = [[], [], []]
    try:
        while True:
            t1 = time.time()
            loss, acc1, acc5 = exe.run(program=test_prog,
                                       fetch_list=test_fetch_list,
                                       use_program_cache=True)
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(loss)
            acc1 = np.mean(acc1)
            acc5 = np.mean(acc5)
            test_info[0].append(loss)
            test_info[1].append(acc1)
            test_info[2].append(acc5)

            if test_batch_id % 100 == 0:
                test_speed = test_batch_size * 1.0 / period
                print("Pass {0},testbatch {1},loss {2}, acc1 {3},acc5 {4},time {5},speed {6}"\
                      .format(pass_id, test_batch_id, "%.5f"%loss,"%.5f"%acc1, "%.5f"%acc5,
                      "%2.2f sec" % period, "%.2f" % test_speed))
            sys.stdout.flush()
            test_batch_id += 1
    except fluid.core.EOFException:
        test_py_reader.reset()
    test_loss = np.array(test_info[0]).mean()
    test_acc1 = np.array(test_info[1]).mean()
    test_acc5 = np.array(test_info[2]).mean()

    print("test_loss {0}, test_acc1 {1}, test_acc5 {2}".format("%.5f"%test_loss,
          "%.5f"%test_acc1, "%.5f"%test_acc5))
    sys.stdout.flush()
    model_path = os.path.join(model_save_dir + '/' + model_name, str(pass_id))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    if num_trainers > 1: 
        fluid.io.save_persistables(exe, model_path, main_program=fleet._origin_program)
    else:
        fluid.io.save_persistables(exe, model_path, main_program=train_prog)

def print_paddle_environments():
    print('--------- Configuration Environments -----------')
    for k in os.environ:
        if "PADDLE_" in k or "FLAGS_" in k:
            print("%s: %s" % (k, os.environ[k]))
    print('------------------------------------------------')

def main():
    args = parser.parse_args()
    # this distributed benchmark code can only support gpu environment.  
    assert args.use_gpu, "only for gpu implementation."
    print_arguments(args)
    print_paddle_environments()
    check_gpu(args.use_gpu)
    train(args)

if __name__ == '__main__':
    main()
