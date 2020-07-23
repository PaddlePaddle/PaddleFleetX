#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
Deep Attention Matching Network
"""
import sys
import os
import six
import numpy as np
import time
import multiprocessing
import paddle
import paddle.fluid as fluid
import reader as reader
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy

try:
    import cPickle as pickle  #python 2
except ImportError as e:
    import pickle  #python 3

from model_check import check_cuda
import utils
import config
import model

def main():
    args = config.parse_args()
    config.print_arguments(args)

    if args.distributed:
        init_dist_env()

    # create executor and place
    # (multi machine mode is different from single machine mode)
    place = create_place(args.distributed)
    exe = create_executor(place)

    # create train network
    train_prog, start_prog = fluid.Program(), fluid.Program()
    with fluid.program_guard(train_prog, start_prog):
        feed, fetch, optimizer = model.build_train_net(args)

    if args.distributed:
        # distributed  optimizer
        optimizer = distributed_optimize(optimizer)    
    optimizer.minimize(fetch[0], start_prog)

    # load data
    with open(args.data_path, 'rb') as f:
        if six.PY2:
            train_data, val_data, test_data = pickle.load(f)
        else:
            train_data, val_data, test_data = pickle.load(
                    f, encoding="bytes")

    # create train dataloader
    # (multi machine mode is different from single machine mode)
    loader = create_train_dataloader(args, train_data, feed, place, args.distributed)
    if args.distributed:
        # be sure to do the following assignment before executing
        # train_prog to specify the program encapsulated by the 
        # distributed policy
        train_prog = fleet.main_program

    # do train
    train(args, train_prog, start_prog, exe, feed, fetch, loader)

    # create test network
    test_prog = fluid.Program()
    with fluid.program_guard(test_prog):
        feed, fetch = model.build_test_net(args)

    # create test dataloader
    loader = create_test_dataloader(args, test_data, feed, place, args.distributed)
    # test on one card
    local_value, local_weight = test(args, test_prog, exe, feed, fetch, loader)

    if args.distributed:
        dist_acc = utils.dist_eval_acc(exe, local_value, local_weight)
        print('[TEST] global_acc1: %.2f' % dist_acc)

def init_dist_env():
    """
    init distributed env by fleet
    """
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)

def create_place(is_distributed):
    """
    decide which device to use based on distributed env
    """
    place_idx = int(os.environ['FLAGS_selected_gpus']) if is_distributed else 0
    return fluid.CUDAPlace(place_idx)

def distributed_optimize(optimizer):
    strategy = DistributedStrategy()
    strategy.fuse_all_reduce_ops = True
    strategy.nccl_comm_num = 2 
    strategy.fuse_elewise_add_act_ops=True
    strategy.fuse_bn_act_ops = True
    return fleet.distributed_optimizer(optimizer, strategy=strategy)

def create_executor(place):
    exe = fluid.Executor(place)
    return exe

def create_train_dataloader(args, data, feed, place, is_distributed):
    return create_dataloader(args, data, feed, place, is_distributed, False)

def create_test_dataloader(args, data, feed, place, is_distributed):
    return create_dataloader(args, data, feed, place, is_distributed, True)

def create_dataloader(args, data, feed, place, is_distributed, is_test):
    data_conf = {
        "batch_size": args.batch_size,
        "max_turn_num": args.max_turn_num,
        "max_turn_len": args.max_turn_len,
        "_EOS_": args._EOS_,
    }
    batch_num = len(data[six.b('y')]) // args.batch_size
    shuffle_data = reader.unison_shuffle(data, seed=None)
    data_batches = reader.build_batches(shuffle_data, data_conf)

    def batch_generator(data_batches, batch_num):
        def generator():
            for index in six.moves.xrange(batch_num):
                yield reader.make_one_batch_input(data_batches, index)
        return generator

    return utils.create_dataloader(
            batch_generator(data_batches, batch_num),
            feed, place, batch_size=args.batch_size,
            is_test=is_test, is_distributed=is_distributed)

def train(args, train_prog, start_prog, exe, feed, fetch, loader):
    exe.run(start_prog)
    for epoch in range(args.num_scan_data):
        for idx, sample in enumerate(loader()):
            ret = exe.run(train_prog, feed=sample, fetch_list=fetch)
            if idx % 1 == 0:
                print('[TRAIN] epoch=%d step=%d loss=%f' % (epoch, idx, ret[0][0]))

def test(args, test_prog, exe, feed, fetch, loader):
    acc_manager = fluid.metrics.Accuracy()
    for idx, sample in enumerate(loader()):
        ret = exe.run(test_prog, feed=sample, fetch_list=fetch)
        """
        scores = np.array(ret[0])
        if idx % 1 == 0:
            for i in six.moves.xrange(args.batch_size):
                print('[TEST] step={} scores={}, label={}'.format(
                    idx, scores[i][0],ctest_batches["label"][idx][i]))
        """
        acc_manager.update(value=ret[0], weight=utils.sample_batch(sample))
        if idx % 1 == 0:
            print('[TEST] step=%d accum_acc1=%.2f' % (idx, acc_manager.eval()))
    print('[TEST] local_acc1: %.2f' % acc_manager.eval())
    return acc_manager.value, acc_manager.weight

if __name__ == '__main__':
    main()
