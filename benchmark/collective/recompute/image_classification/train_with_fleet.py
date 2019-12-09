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

from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker

num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
trainer_id = int(os.environ.get('PADDLE_TRAINER_ID'))

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

import paddle
import paddle.fluid as fluid
import reader
from utils import *
import models
from build_model import create_model


def build_program(is_train, main_prog, startup_prog, args, dist_strategy):
    """build program, and add grad op in program accroding to different mode

    Args:
        is_train: mode: train or test
        main_prog: main program
        startup_prog: strartup program
        args: arguments

    Returns : 
        train mode: [Loss, global_lr, py_reader]
        test mode: [Loss, py_reader]
    """
    if args.model.startswith('EfficientNet'):
        is_test = False if is_train else True
        override_params = {"drop_connect_rate": args.drop_connect_rate}
        padding_type = args.padding_type
        model = models.__dict__[args.model](is_test=is_test, override_params=override_params, padding_type=padding_type)
    else:
        model = models.__dict__[args.model]()
    with fluid.program_guard(main_prog, startup_prog):
        if args.random_seed:
            main_prog.random_seed = args.random_seed
            startup_prog.random_seed = args.random_seed
        with fluid.unique_name.guard():
            py_reader, loss_out = create_model(model, args, is_train)
            # add backward op in program
            if is_train:
                optimizer = create_optimizer(args)
                global_lr = optimizer._global_learning_rate()
                #if args.use_recompute:
                #    optimizer = fluid.optimizer.RecomputeOptimizer(optimizer)
                #    #print(model.checkpoints)
                #    optimizer._set_checkpoints(model.checkpoints)
                if args.use_recompute:
                    print("Recompute!!!")
                    print("checkpoints: ", model.checkpoints)
                    dist_strategy.recompute_checkpoints = model.checkpoints

                dist_optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
                avg_cost = loss_out[0]
                _, param_grads = dist_optimizer.minimize(avg_cost)

                global_lr.persistable = True
                loss_out.append(global_lr)
                if args.use_ema:
                    global_steps = fluid.layers.learning_rate_scheduler._decay_step_counter()
                    ema = ExponentialMovingAverage(args.ema_decay, thres_steps=global_steps)
                    ema.update()
                    loss_out.append(ema)
            loss_out.append(py_reader)
    return loss_out

def validate(args, test_py_reader, exe, test_prog, test_fetch_list, pass_id, train_batch_metrics_record):
    test_batch_time_record = []
    test_batch_metrics_record = []
    test_batch_id = 0
    test_py_reader.start()
    try:
        while True:
            t1 = time.time()
            test_batch_metrics = exe.run(program=test_prog,
                                         fetch_list=test_fetch_list)
            t2 = time.time()
            test_batch_elapse = t2 - t1
            test_batch_time_record.append(test_batch_elapse)

            test_batch_metrics_avg = np.mean(
                np.array(test_batch_metrics), axis=1)
            test_batch_metrics_record.append(test_batch_metrics_avg)

            print_info(pass_id, test_batch_id, args.print_step,
                       test_batch_metrics_avg, test_batch_elapse, "batch")
            sys.stdout.flush()
            test_batch_id += 1

    except fluid.core.EOFException:
        test_py_reader.reset()
    #train_epoch_time_avg = np.mean(np.array(train_batch_time_record))
    train_epoch_metrics_avg = np.mean(
        np.array(train_batch_metrics_record), axis=0)

    test_epoch_time_avg = np.mean(np.array(test_batch_time_record))
    test_epoch_metrics_avg = np.mean(
        np.array(test_batch_metrics_record), axis=0)

    print_info(pass_id, 0, 0,
               list(train_epoch_metrics_avg) + list(test_epoch_metrics_avg),
               test_epoch_time_avg, "epoch")

def train(args):
    """Train model
    
    Args:
        args: all arguments.    
    """
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    build_strategy = fluid.compiler.BuildStrategy()

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 1 # each card is a process
    # exec_strategy.num_threads = fluid.core.get_cuda_device_count()
    exec_strategy.num_iteration_per_drop_scope = 10

    dist_strategy = DistributedStrategy()
    dist_strategy.exec_strategy = exec_strategy
    dist_strategy.nccl_comm_num = 1
    dist_strategy.enable_sequential_execution = True 
    if args.use_recompute:
        print("use Recompute!!!")
        dist_strategy.forward_recompute = True
    
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)

    train_out = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args,
        dist_strategy=dist_strategy)

    train_py_reader = train_out[-1]
    if args.use_ema:
        train_fetch_vars = train_out[:-2]
        ema = train_out[-2]
    else:
        train_fetch_vars = train_out[:-1]

    train_fetch_list = [var.name for var in train_fetch_vars]

    train_prog = fleet.main_program

    test_out = build_program(
        is_train=False,
        main_prog=test_prog,
        startup_prog=startup_prog,
        args=args,
        dist_strategy=dist_strategy)
 
    test_py_reader = test_out[-1]
    test_fetch_vars = test_out[:-1]

    test_fetch_list = [var.name for var in test_fetch_vars]

    #Create test_prog and set layers' is_test params to True
    test_prog = test_prog.clone(for_test=True)

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    print("gpu_id: ", gpu_id)
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    #init model by checkpoint or pretrianed model.
    init_model(exe, args, train_prog)

    # TODO(mapingshuo) for fleet training, the trainers in diff gpu cards
    # runs in different process, same shuffle seed need to be set for each
    # card. It is important if we want to run the models to converge.

    train_reader = reader.train(settings=args)
    train_reader = paddle.batch(
        train_reader,
        batch_size=int(args.batch_size),
        drop_last=True)

    test_reader = reader.val(settings=args)
    test_reader = paddle.batch(
        test_reader, batch_size=args.test_batch_size, drop_last=True)

    train_py_reader.decorate_sample_list_generator(train_reader, place)
    test_py_reader.decorate_sample_list_generator(test_reader, place)

    #compiled_train_prog = best_strategy_compiled(args, train_prog, train_fetch_vars[0])
    train_speed_list = []
    for pass_id in range(args.num_epochs):
        begin_time = time.time()
        train_batch_id = 0
        train_batch_time_record = []
        train_batch_metrics_record = []
        train_begin=time.time()
        train_py_reader.start()

        try:
            while True:
                t1 = time.time()
                #train_batch_metrics = exe.run(compiled_train_prog, fetch_list=train_fetch_list, use_program_cache=True)
                train_batch_metrics = exe.run(train_prog, fetch_list=train_fetch_list, use_program_cache=True)
                t2 = time.time()
                train_batch_elapse = t2 - t1
                train_batch_time_record.append(train_batch_elapse)
                train_batch_metrics_avg = np.mean(
                    np.array(train_batch_metrics), axis=1)
                train_batch_metrics_record.append(train_batch_metrics_avg)

                print_info(pass_id, train_batch_id, args.print_step,
                           train_batch_metrics_avg, train_batch_elapse, "batch")
                sys.stdout.flush()
                train_batch_id += 1

        except fluid.core.EOFException:
            train_py_reader.reset()
        print("epoch {}, used {} seconds".format(pass_id, time.time()-begin_time))

        if args.use_ema:
            print('ExponentialMovingAverage validate start...')
            with ema.apply(exe):
                validate(args, test_py_reader, exe, test_prog, test_fetch_list, pass_id, train_batch_metrics_record)
            print('ExponentialMovingAverage validate over!')

        #validate(args, test_py_reader, exe, test_prog, test_fetch_list, pass_id, train_batch_metrics_record)
        #For now, save model per epoch.
        if pass_id % args.save_step == 0:
            save_model(args, exe, train_prog, pass_id)
        train_end=time.time()
        train_speed = (train_batch_id * args.batch_size) / (train_end - train_begin)
        train_speed_list.append(train_speed) 

    # save train log
    if trainer_id == 0: 
        if not os.path.isdir("./benchmark_logs/"):
            os.makedirs("./benchmark_logs/")
            with open("./benchmark_logs/log_%d" % trainer_id, 'w') as f:
                result = dict()
                result['1'] = 1 #np.mean(train_speed_list) * num_trainers
                result['14'] = 32 #args.batch_size
                print(str(result))
                f.writelines(str(result))


def main():
    args = parse_args()
    print_arguments(args)
    check_args(args)
    train(args)


if __name__ == '__main__':
    main()
