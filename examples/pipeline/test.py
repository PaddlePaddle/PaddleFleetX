from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy
import argparse
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import reader
import cifar_reader
import os

parser = argparse.ArgumentParser(description="Evaluate Resnet.")
parser.add_argument('--data_dir',
                    type=str,
                    default="./",
                    help="Data directory")
parser.add_argument('--data_file',
                    type=str,
                    default="val.txt",
                    help="Data file")
parser.add_argument('--dataset',
                    type=str,
                    default="ImageNet",
                    help="Dataset to use, ImageNet or cifar10.")
parser.add_argument('--passid',
                    type=int,
                    default=0,
                    help="Pass id")
parser.add_argument('--model_dir',
                    type=str,
                    default='./saved_model',
                    help="Dir for saved model")
args = parser.parse_args()

if args.dataset == "ImageNet":
    class_num = 1000
    image_shape = '3,224,224'
elif args.dataset == "cifar10":
    class_num = 10
    image_shape = '3,32,32'
else:
    raise ValueError("Unknown dataset: {}.".format(args.dataset))

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
                                   is_test=True)

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
    if layers >= 50:
        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = bottleneck_block(input=conv,
                                        num_filters=num_filters[block],
                                        stride=2 if i == 0 and block != 0 else 1)
        pool = fluid.layers.pool2d(input=conv,
                                   pool_size=7,
                                   pool_type='avg',
                                   global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(input=pool,
                              size=class_dim,
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv, stdv)))
    else:
        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = basic_block(input=conv,
                                   num_filters=num_filters[block],
                                   stride=2 if i == 0 and block != 0 else 1,
                                   is_first=block==i==0)
        pool = fluid.layers.pool2d(input=conv,
                                   pool_size=7,
                                   pool_type='avg',
                                   global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(input=pool,
                              size=class_dim,
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv, stdv)))
    return out


image_shape = [int(s) for s in image_shape.split(',')]
image = fluid.layers.data(name="image", shape=image_shape, dtype="float32")
label = fluid.layers.data(name="label", shape=[1], dtype="int64")
fc = build_network(image, layers=50, class_dim=class_num)
out, prob = fluid.layers.softmax_with_cross_entropy(logits=fc, label=label, return_softmax=True)
loss = fluid.layers.mean(out)
acc_top1 = fluid.layers.accuracy(input=prob, label=label, k=1)
acc_top5 = fluid.layers.accuracy(input=prob, label=label, k=5)

startup_prog = fluid.default_startup_program()
test_prog = fluid.default_main_program().clone(for_test=True)

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(startup_prog)

if args.dataset == "ImageNet":
    test_reader = paddle.batch(reader.test(args.data_dir), batch_size=100)
elif args.dataset == "cifar10":
    filepath = os.path.join(args.data_dir, args.data_file)
    test_reader = paddle.batch(cifar_reader.test10(filepath), batch_size=100)
else:
    raise ValueError("Unknown dataset: {}.".format(args.dataset))

feeder = fluid.DataFeeder(place=place,
                          feed_list=['image', 'label'],
                          program=test_prog)
fetch_list = [loss.name, acc_top1.name, acc_top5.name]

model_dir = os.path.join(args.model_dir, str(args.passid))
fluid.io.load_persistables(exe,
                           dirname=model_dir,
                           main_program=test_prog)

total_acc1 = []
total_acc5 = []
total_num = 0
for batch_id, data in enumerate(test_reader()):
    [loss, acc1, acc5] = exe.run(test_prog,
                                 use_program_cache=True,
                                 feed=feeder.feed(data),
                                 fetch_list=fetch_list)
    data_num = len(data)
    total_num += data_num
    total_acc1.append(acc1[0] * data_num)
    total_acc5.append(acc5[0] * data_num)
    print("batch_id: {}, loss: {:.6f}, acc1: {:.6f}, acc5: {:.6f}".format(
                batch_id, loss[0], acc1[0], acc5[0]))
acc1 = numpy.sum(total_acc1) / total_num
acc5 = numpy.sum(total_acc5) / total_num
print("pass: {}, avg acc1: {:.6f}, avg acc5: {:.6f}.".format(
           args.passid, acc1[0], acc5[0]))
