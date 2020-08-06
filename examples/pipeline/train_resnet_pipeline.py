from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy
import os
import argparse
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import reader
import cifar_reader

parser = argparse.ArgumentParser(description="Resnet with pipeline parallel.")
parser.add_argument('--data_dir',
                    type=str,
                    default="./",
                    help="Data directory")
parser.add_argument('--dataset',
                    type=str,
                    default="ImageNet",
                    help="ImageNet or cifar10")
parser.add_argument('--random_seed',
                    type=int,
                    default=0,
                    help="Random seed for program.")
parser.add_argument('--data_file',
                    type=str,
                    default="train.txt",
                    help="Data file.")
parser.add_argument('--microbatch_size',
                    type=int,
                    default=32,
                    help="Micro-batch size")
parser.add_argument('--microbatch_num',
                    type=int,
                    default=1,
                    help="Number of microbatchs")
parser.add_argument('--passid',
                    type=int,
                    default=0,
                    help="Pass id")
parser.add_argument('--use_fixed_lr',
                    default=False,
                    action='store_true',
                    help="Use fixed lr or not")
parser.add_argument('--model_dir',
                    type=str,
                    default='./saved_model',
                    help="Dir for the saved model")

args = parser.parse_args()

if args.dataset == "ImageNet":
    class_num = 1000
    image_shape = "3,224,224"
elif args.dataset == "cifar10":
    class_num = 10
    image_shape = "3,32,32"
else:
    raise ValueError("Unknown dataset: {}".format(args.dataset))

def conv_bn_layer(input,
                  num_filters,
                  filter_size,
                  stride=1,
                  groups=1,
                  act=None):
    conv = fluid.layers.conv2d(input=input,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=stride,
                               padding=(filter_size-1) // 2,
                               groups=groups,
                               act=None,
                               bias_attr=False)
    return fluid.layers.batch_norm(input=conv, 
                                   act=act,
                                  )

def shortcut(input,
             ch_out,
             stride,
             is_first):
    ch_in = input.shape[1]
    if ch_in != ch_out or stride != 1 or is_first == True:
        return conv_bn_layer(input,
                             ch_out,
                             1,
                             stride)
    else:
        return input

def bottleneck_block(input,
                     num_filters,
                     stride):
    conv0 = conv_bn_layer(input=input,
                          num_filters=num_filters,
                          filter_size=1,
                          act='relu')
    conv1 = conv_bn_layer(input=conv0,
                          num_filters=num_filters,
                          filter_size=3,
                          stride=stride,
                          act='relu')
    conv2 = conv_bn_layer(input=conv1,
                          num_filters=num_filters*4,
                          filter_size=1,
                          act=None)

    short = shortcut(input,
                     num_filters*4,
                     stride,
                     is_first=False)

    return fluid.layers.elementwise_add(x=short,
                                        y=conv2,
                                        act='relu')

def basic_block(input,
                num_filters,
                stride,
                is_first):
    conv0 = conv_bn_layer(input=input,
                          num_filters=num_filters,
                          filter_size=3,
                          act='relu',
                          stride=stride)
    conv1 = conv_bn_layer(input=conv0,
                          num_filters=num_filters,
                          filter_size=3,
                          act=None)
    short = shortcut(input,
                     num_filters,
                     stride,
                     is_first)
    return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')


def build_network(input,
                  layers=50,
                  class_dim=1000):
    supported_layers = [18, 34, 50, 101, 152]
    assert layers in supported_layers
    depth = None
    if layers == 18:
        depth = [2, 2, 2, 2]
    elif layers == 34 or layers == 50:
        depth = [3, 4, 6, 3]
    elif layers == 101:
        depth = [3, 4, 23, 3]
    elif layers == 152:
        depth = [3, 8, 36, 3]
    num_filters = [64, 128, 256, 512]
    offset = 0
    with fluid.device_guard("gpu:%d"%(offset)):
        conv = conv_bn_layer(input=input,
                             num_filters=64,
                             filter_size=7,
                             stride=2,
                             act='relu')
        conv = fluid.layers.pool2d(input=conv,
                                   pool_size=3,
                                   pool_stride=2,
                                   pool_padding=1,
                                   pool_type='max')
    offset += 1
    if layers >= 50:
        for block in range(len(depth)):
            with fluid.device_guard("gpu:%d"%(offset)):
                for i in range(depth[block]):
                    conv = bottleneck_block(
                            input=conv,
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1)
            offset += 1

        with fluid.device_guard("gpu:%d"%(offset)):
            pool = fluid.layers.pool2d(input=conv,
                                       pool_size=7,
                                       pool_type='avg',
                                       global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                    input=pool,
                    size=class_dim,
                    param_attr=fluid.param_attr.ParamAttr(
                        initializer=fluid.initializer.Uniform(-stdv, stdv)))
        #offset += 1
    else:
        for block in range(len(depth)):
            with fluid.device_guard("gpu:%d"%(offset)):
                for i in range(depth[block]):
                    conv = basic_block(input=conv,
                                       num_filters=num_filters[block],
                                       stride=2 if i == 0 and block != 0 else 1,
                                       is_first=block==i==0)
            offset += 1
        with fluid.device_guard("gpu:%d"%(offset)):
            pool = fluid.layers.pool2d(input=conv,
                                       pool_size=7,
                                       pool_type='avg',
                                       global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                    input=pool,
                    size=class_dim,
                    param_attr=fluid.param_attr.ParamAttr(
                        initializer=fluid.initializer.Uniform(-stdv, stdv)))
        #offset += 1
    return out, offset

if args.random_seed:
    fluid.default_main_program().random_seed = args.random_seed
    fluid.default_startup_program().random_seed = args.random_seed

with fluid.device_guard("gpu:0"):
    image_shape = [int(s) for s in image_shape.split(',')]
    image = fluid.layers.data(name="image",
                              shape=image_shape,
                              dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[image, label],
            capacity=64,
            use_double_buffer=True,
            iterable=False)
    fc, offset = build_network(image, layers=50, class_dim=class_num) 
with fluid.device_guard("gpu:%d"%(offset)):
    out, prob = fluid.layers.softmax_with_cross_entropy(logits=fc,
                                                        label=label,
                                                        return_softmax=True)
    loss = fluid.layers.mean(out)
    acc_top1 = fluid.layers.accuracy(input=prob, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=prob, label=label, k=5)

mini_batch_size = args.microbatch_num * args.microbatch_size
print("mini_batch_size:", mini_batch_size)
lr_val = None
if args.use_fixed_lr:
    lr_val = 0.1
else:
    base_lr = 0.1
    passes = [30, 60, 80, 90]
    total_images = 1281167
    steps_per_pass = int(math.ceil(total_images/mini_batch_size))
    bd = [steps_per_pass * p for p in passes]
    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
    lr_val = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
opt = fluid.optimizer.Momentum(lr_val, momentum=0.9)
#opt = fluid.contrib.mixed_precision.decorate(opt,
#                                             init_loss_scaling=128)
strategy = paddle.fleet.DistributedStrategy()
strategy.pipeline = True
strategy.pipeline_configs = {'micro_batch': args.microbatch_num}
opt = fleet.distributed_optimizer(opt, strategy=strategy)

opt.minimize(loss)

place = fluid.CUDAPlace(0)
if args.dataset == "ImageNet":
    train_reader = reader.train(args.data_dir,
                                file_list=args.data_file,
                                shuffle=False if args.random_seed else True,
                                pass_id_as_seed=args.passid)
elif args.dataset == "cifar10":
    filepath = os.path.join(args.data_dir, args.data_file)
    train_reader = cifar_reader.train10(filepath)
else:
    raise ValueError("Unknown dataset: {}".format(args.dataset))
data_loader.set_sample_generator(train_reader,
                                 batch_size=args.microbatch_size)

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

assert args.model_dir

for pass_id in range(args.passid, 90):
    if pass_id > 0:
        model_dir = os.path.join(args.model_dir, str(pass_id-1))
        print('model_dir:', model_dir)
        fluid.io.load_persistables(exe,
                                   dirname=model_dir,
                                   main_program=fluid.default_main_program())
    data_loader.start()
    exe.train_from_dataset(fluid.default_main_program(), debug=True)
    data_loader.reset()
    print("trained pass_id: ", args.passid)
    save_dir = os.path.join(args.model_dir, str(passid))
    fluid.io.save_persistables(
        exe,
        dirname=save_dir,
        main_program=fluid.default_main_program())
