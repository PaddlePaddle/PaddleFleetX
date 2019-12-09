#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
"""Contains common utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils.util
import numpy as np
import six

def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-------------  Configuration Arguments -------------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%25s : %s" % (arg, value))
    print("----------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)

def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not fluid.is_compiled_with_cuda():
            logger.error(err)
            sys.exit(1)
    except Exception as e:
        pass

def get_median(data):
    data = sorted(data)
    size = len(data)
    if size == 2:
        median = data[size-1]
    elif size % 2 == 0:
        median = (data[size//2]+data[size//2-1])/2
    elif size % 2 == 1:
        median = data[(size-1)//2]
    return median


def create_data_loader(is_train, args):
    """create data_loader

    Usage:
        Using mixup process in training, it will return 5 results, include data_loader, image, y_a(label), y_b(label) and lamda, or it will return 3 results, include data_loader, image, and label.

    Args:
        is_train: mode
        args: arguments

    Returns:
        data_loader and the input data of net,
    """
    image_shape = args.image_shape
    feed_image = fluid.data(
        name="feed_image",
        shape=[None] + image_shape,
        dtype="float32",
        lod_level=0)

    feed_label = fluid.data(
        name="feed_label", shape=[None, 1], dtype="int64", lod_level=0)
    feed_y_a = fluid.data(
        name="feed_y_a", shape=[None, 1], dtype="int64", lod_level=0)

    if is_train and args.use_mixup:
        feed_y_b = fluid.data(
            name="feed_y_b", shape=[None, 1], dtype="int64", lod_level=0)
        feed_lam = fluid.data(
            name="feed_lam", shape=[None, 1], dtype="float32", lod_level=0)

        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[feed_image, feed_y_a, feed_y_b, feed_lam],
            capacity=64,
            use_double_buffer=True,
            iterable=True)
        return data_loader, [feed_image, feed_y_a, feed_y_b, feed_lam]
    else:
        if args.use_dali:
            return None, [feed_image, feed_label]

        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[feed_image, feed_label],
            capacity=64,
            use_double_buffer=True,
            iterable=True)

        return data_loader, [feed_image, feed_label]

def parse_args():
    """Add arguments

    Returns: 
        all training args
    """
    # yapf: disable
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)

    add_arg('batch_size',       int,   32,                   "Minibatch size per device.")
    add_arg('total_images',     int,   1281167,              "Training image number.")
    add_arg('num_epochs',       int,   120,                  "number of epochs.")
    add_arg('class_dim',        int,   1000,                 "Class number.")
    add_arg('image_shape',      str,   "3,224,224",          "input image size")
    add_arg('model_save_dir',   str,   "output",             "model save directory")
    add_arg('with_mem_opt',     bool,  False,                "Whether to use memory optimization or not.")
    add_arg('with_inplace',     bool,  False,                "Whether to use inplace memory optimization.")
    add_arg('pretrained_model', str,   None,                 "Whether to use pretrained model.")
    add_arg('checkpoint',       str,   None,                 "Whether to resume checkpoint.")
    add_arg('lr',               float, 0.1,                  "set learning rate.")
    add_arg('lr_strategy',      str,   "piecewise_decay",    "Set the learning rate decay strategy.")
    add_arg('model',            str,   "SE_ResNeXt50_32x4d", "Set the network to use.")
    add_arg('data_dir',         str,   "./data/ILSVRC2012/",  "The ImageNet dataset root dir.")
    add_arg('fp16',             bool,  False,                "Enable half precision training with fp16." )
    add_arg('use_dali',             bool,  False,            "use DALI for preprocess or not." )
    add_arg('data_format',      str,   "NCHW",               "Tensor data format when training.")
    add_arg('scale_loss',       float, 1.0,                  "Scale loss for fp16." )
    add_arg('use_dynamic_loss_scaling',     bool,   True,    "Whether to use dynamic loss scaling.")
    add_arg('l2_decay',         float, 1e-4,                 "L2_decay parameter.")
    add_arg('momentum_rate',    float, 0.9,                  "momentum_rate.")
    add_arg('use_label_smoothing',      bool,      False,        "Whether to use label_smoothing or not")
    add_arg('label_smoothing_epsilon',      float,     0.2,      "Set the label_smoothing_epsilon parameter")
    add_arg('lower_scale',      float,     0.08,      "Set the lower_scale in ramdom_crop")
    add_arg('lower_ratio',      float,     3./4.,      "Set the lower_ratio in ramdom_crop")
    add_arg('upper_ratio',      float,     4./3.,      "Set the upper_ratio in ramdom_crop")
    add_arg('resize_short_size',      int,     256,      "Set the resize_short_size")
    add_arg('use_mixup',      bool,      False,        "Whether to use mixup or not")
    add_arg('mixup_alpha',      float,     0.2,      "Set the mixup_alpha parameter")
    add_arg('is_distill',       bool,  False,        "is distill or not")
    add_arg('profile',             bool,  False,                "Enable profiler or not." )
    add_arg('print_program_desc',             bool,  False,                "Enable print program desc or not." )
    add_arg('fetch_steps',      int,  10,                "Enable profiler or not." )

    add_arg('do_test',          bool,  False,                 "Whether do test every epoch.")
    add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
    add_arg('fuse', bool, False,                      "Whether to use tensor fusion.")
    add_arg('nccl_comm_num',        int,  1,                  "nccl comm num")
    add_arg("use_hierarchical_allreduce",     bool,   False,   "Use hierarchical allreduce or not.")
    add_arg('num_threads',        int,  1,                   "Use num_threads to run the fluid program.")
    add_arg('num_iteration_per_drop_scope', int,    100,      "Ihe iteration intervals to clean up temporary variables.")
    add_arg('benchmark_test',          bool,  True,                 "Whether to use print benchmark logs or not.")

    add_arg('use_dgc',           bool,  False,          "Whether use DGCMomentum Optimizer or not")
    add_arg('rampup_begin_step', int,   5008,           "The beginning step from which dgc is implemented.")

    # yapf: enable
    args = parser.parse_args()
    return args
