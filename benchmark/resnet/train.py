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
import argparse
import six
import fleetx as X
import numpy as np
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import time


def parse_args():
    parser = argparse.ArgumentParser("resnet")

    parser.add_argument(
        "--epoch", type=int, default=90, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Minibatch size per device")
    parser.add_argument(
        "--lr", type=float, default=0.1, help="base learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum value")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay")

    parser.add_argument(
        "--use_dgc", type=bool, default=False, help="Whether use dgc(Deep Gradient Compression)")
    parser.add_argument(
        "--use_amp", type=bool, default=False, help="Whether use amp(Automatic Mixed Precision)")

    args = parser.parse_args()
    return args


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in six.iteritems(vars(args)):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


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
    loader = model.load_imagenet_from_file("./ImageNet/train.txt", batch_size=args.batch_size)
    test_loader = model.load_imagenet_from_file("./ImageNet/val.txt", phase='val', batch_size=args.batch_size)
    
    optimizer = fluid.optimizer.Momentum(
        learning_rate=args.lr,
        momentum=args.momentum,
        regularization=fluid.regularizer.L2Decay(args.weight_decay))
    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    optimizer.minimize(model.loss)
    
    place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    exe = fluid.Executor(place)
    
    exe.run(model.startup_prog)
    print("============start training============")
    for epoch_id in range(args.epoch):
        fetch_vars = [model.loss] + model.target
        total_time = 0
        for step, data in enumerate(loader()):
            start_time = time.time()
            loss, acc1, acc5 = exe.run(model.main_prog,
                                       feed=data,
                                       fetch_list=fetch_vars,
                                       use_program_cache=True)
            end_time = time.time()
            total_time += (end_time - start_time)
            if step % 10 == 0:
                print("Train epoch %d, step %d, loss %f, acc1 %f, acc5 %f, total time cost %f, average speed %f"
                    % (epoch_id, step, loss[0], acc1[0], acc5[0],  total_time, (step+1) * args.batch_size / total_time))
    
        if rank == 0 and (epoch_id % 5 == 0 or epoch_id >= args.epoch - 10):
            for j, test_data in enumerate(test_loader()):
                acc1, acc5, loss = exe.run(model.test_prog,
                                           feed=test_data,
                                           fetch_list=fetch_vars,
                                           use_program_cache=True)
                if j % 10 == 0:
                    print("Test epoch %d, step %d, loss %f, acc1 %f, acc5 %f"
                        % (epoch_id, j, loss, acc1[0], acc5[0]))


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)

    main(args)
