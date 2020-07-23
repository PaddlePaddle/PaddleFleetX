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

# 参数设置
parser = argparse.ArgumentParser()
# 多卡分布式训练标记，用以显示区别分布式训练和单卡训练在代码上的差异，便于读者辨识。注意：单卡训练时不要指定此配置
parser.add_argument('--distributed', action='store_true', default=False)
# 批量大小
parser.add_argument('--batch_size', type=int, default=16)
# 训练轮次
parser.add_argument('--epoch_num', type=int, default=2)
# 学习率
parser.add_argument('--learning_rate', type=float, default=2e-3)
args = parser.parse_args()


def main():
    '''
    主训练函数，呈现常见的分布式训练配置步骤
    '''
    if args.distributed:
        # 如果存在分布式标记则初始化分布式环境
        init_dist_env()

    # 创建执行器和其依赖的设备。其中Place（设备）创建与分布式标记相关
    place = create_place(args.distributed)
    exe = create_executor(place)

    # 创建训练相关网络，含初始化网络start_prog和训练网络train_prog
    train_prog, start_prog = fluid.Program(), fluid.Program()
    with fluid.program_guard(train_prog, start_prog):
        feed, fetch = model.build_train_net()

    # 创建优化器并执行优化，其中优化器创建与分布式标记相关，需要进行分布式策略封装
    optimizer = create_optimizer(args.distributed)
    optimizer.minimize(fetch[0], start_prog)

    # 创建训练数据加载器，与分布式标记相关，决定此程序训练用的训练数据分片
    loader = create_train_dataloader(feed, place, args.distributed)
    if args.distributed:
        # 如果存在分布式标记，务必在执行train_prog前做如下赋值操作，指定分布式策略封装后的program。后续会对此约束进行优化
        train_prog = fleet.main_program
    # 执行训练
    train(train_prog, start_prog, exe, feed, fetch, loader)

    # 创建测试网络
    test_prog = fluid.Program()
    with fluid.program_guard(test_prog):
        feed, fetch = model.build_test_net()

    # 创建测试数据加载器，同样依赖分布式标记，决定测试数据分片
    loader = create_test_dataloader(feed, place, args.distributed)
    # 执行单卡测试，对本地的数据计算评估指标
    local_value, local_weight = test(test_prog, exe, feed, fetch, loader)

    if args.distributed:
        # 如果有分布式标记，还需要汇总其他worker的评估指标
        dist_acc = utils.dist_eval_acc(exe, local_value, local_weight)
        print('[TEST] global_acc1: %.2f' % dist_acc)


def init_dist_env():
    '''
    使用Paddle Fleet初始化分布式环境
    '''
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)


def create_place(is_distributed):
    '''
    根据分布式训练环境变量决定使用的设备号
    '''
    place_idx = int(os.environ['FLAGS_selected_gpus']) if is_distributed else 0
    return fluid.CUDAPlace(place_idx)


def distributed_optimize(optimizer):
    '''
    分布式训练相关配置
    '''
    strategy = DistributedStrategy()
    strategy.fuse_all_reduce_ops = True
    strategy.nccl_comm_num = 2 
    strategy.fuse_elewise_add_act_ops=True
    strategy.fuse_bn_act_ops = True
    return fleet.distributed_optimizer(optimizer, strategy=strategy)


def create_executor(place):
    '''
    创建执行器
    '''
    exe = fluid.Executor(place)
    return exe


def create_optimizer(is_distributed):
    '''
    创建优化器，且根据分布式标签决定是否进行分布式封装
    '''
    optimizer = fluid.optimizer.SGD(learning_rate=args.learning_rate)
    if is_distributed:
        optimizer = distributed_optimize(optimizer)    
    return optimizer


def create_train_dataloader(feed, place, is_distributed):
    '''
    如果本地有训练数据则直接应用，否则需要外网下载数据
    '''
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
    '''
    如果本地有测试数据则直接应用，否则需要外网下载数据
    '''
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
    '''
    常规训练代码，先执行初始化网络完成参数初始化，然后逐批量训练epoch_num轮数
    '''
    exe.run(start_prog)
    for epoch in range(args.epoch_num):
        for idx, sample in enumerate(loader()):
            ret = exe.run(train_prog, feed=sample, fetch_list=fetch)
            if idx % 100 == 0:
                print('[TRAIN] epoch=%d step=%d loss=%f' % (epoch, idx, ret[0][0]))


def test(test_prog, exe, feed, fetch, loader):
    '''
    以正确率评估为例，局部累积正确样本数（acc_magener.value）和样本总数（acc_manager.weight）,
    然后汇总全局正确样本数和全局样本数后即可通过两者的商算出全局的正确率
    '''
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

