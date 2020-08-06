from __future__ import print_function
import os
import numpy as np
import argparse

import paddle
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.base import role_maker

import model
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--distributed', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch_num', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=2e-3)
args = parser.parse_args()


def main():
    if args.distributed:
        init_dist_env()

    place = create_place(args.distributed)
    exe = create_executor(place)

    train_prog, start_prog = fluid.Program(), fluid.Program()
    with fluid.program_guard(train_prog, start_prog):
        feed, fetch = model.build_train_net(args.distributed)

    optimizer = create_optimizer(args.distributed)
    optimizer.minimize(fetch[0], start_prog)

    loader = create_train_dataloader(feed, place, args.distributed)
    if args.distributed:
        train_prog = fleet.main_program
    train(train_prog, start_prog, exe, feed, fetch, loader)

    test_prog = fluid.Program()
    with fluid.program_guard(test_prog):
        feed, fetch = model.build_test_net()

    loader = create_test_dataloader(feed, place, args.distributed)
    local_value, local_weight = test(test_prog, exe, feed, fetch, loader)

    if args.distributed:
        dist_acc = utils.dist_eval_acc(exe, local_value, local_weight)
        print('[TEST] global_acc1: %.2f' % dist_acc)


def init_dist_env():
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)


def create_place(is_distributed):
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


def create_optimizer(is_distributed):
    optimizer = fluid.optimizer.SGD(learning_rate=args.learning_rate)
    if is_distributed:
        optimizer = distributed_optimize(optimizer)    
    return optimizer


def create_train_dataloader(feed, place, is_distributed):
    train_data_path = 'dataset/train-images-idx3-ubyte.gz'
    train_label_path = 'dataset/train-labels-idx1-ubyte.gz'
    if os.path.exists(train_data_path) and os.path.exists(train_label_path):
        reader = paddle.dataset.mnist.reader_creator(train_data_path,
                train_label_path, 100)
    else:
        reader = paddle.dataset.mnist.train()
    return utils.create_dataloader(reader, feed, place,
            batch_size=args.batch_size, is_test=False, is_distributed=is_distributed)


def create_test_dataloader(feed, place, is_distributed):
    test_data_path = 'dataset/t10k-images-idx3-ubyte.gz'
    test_label_path = 'dataset/t10k-labels-idx1-ubyte.gz'
    if os.path.exists(test_data_path) and os.path.exists(test_label_path):
        reader = paddle.dataset.mnist.reader_creator(test_data_path,
                test_label_path, 100)
    else:
        reader = paddle.dataset.mnist.test()
    return utils.create_dataloader(reader, feed, place,
            batch_size=args.batch_size, is_test=True, is_distributed=is_distributed)


def train(train_prog, start_prog, exe, feed, fetch, loader):
    exe.run(start_prog)
    for epoch in range(args.epoch_num):
        for idx, sample in enumerate(loader()):
            ret = exe.run(train_prog, feed=sample, fetch_list=fetch)
            if idx % 100 == 0:
                print('[TRAIN] epoch=%d step=%d loss=%f' % (epoch, idx, ret[0][0]))


def test(test_prog, exe, feed, fetch, loader):
    acc_manager = fluid.metrics.Accuracy()
    for idx, sample in enumerate(loader()):
        ret = exe.run(test_prog, feed=sample, fetch_list=fetch)
        acc_manager.update(value=ret[0], weight=utils.sample_batch(sample))
        if idx % 100 == 0:
            print('[TEST] step=%d accum_acc1=%.2f' % (idx, acc_manager.eval()))
    print('[TEST] local_acc1: %.2f' % acc_manager.eval())
    return acc_manager.value, acc_manager.weight


if __name__ == '__main__':
    main()

