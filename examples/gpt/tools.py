# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import yaml
import paddle
import paddle.distributed as dist
import argparse
from fleetx.datasets.gpt import create_pretrained_dataset, get_train_data_file


def process_batch_size(args):
    """
    process_batch_size for hybrid parallel
    """
    if args.global_batch_size is None and args.local_batch_size is None:
        raise ValueError(
            "global_batch_size or local_batch_size should be set.")
    elif args.global_batch_size is not None and args.local_batch_size is not None:
        assert args.global_batch_size // args.local_batch_size == (args.dp_degree *
            args.sharding_degree), "global_batch_size[{}] should be divided by local_batch_size[{}] "\
            "when dp_degree is [{}] and sharding_degree is [{}]".format(args.global_batch_size,
            args.local_batch_size, args.dp_degree, args.sharding_degree)
    elif args.global_batch_size is not None and args.local_batch_size is None:
        assert args.global_batch_size % (args.dp_degree * args.sharding_degree) == 0, \
            "global_batch_size[{}] should be divided by dp_degree[{}] times sharding_degree[{}]"\
            .format(args.global_batch_size, args.dp_degree, args.sharding_degree)
        args.local_batch_size = args.global_batch_size // (
            args.dp_degree * args.sharding_degree)
    else:
        args.global_batch_size = args.local_batch_size * args.dp_degree * args.sharding_degree
    assert args.local_batch_size % args.micro_batch_size == 0


def model_size(args):
    """
    get model size for transformer
    """
    l = args.num_layers
    h = args.hidden_size
    V = args.vocab_size
    S = args.max_seq_len
    P = 12 * l * h * h * (1 + 13 / (12 * h) + (V + S) / (12 * l * h))

    print('Model Size: {:.2f} B'.format(P / 1000.0 / 1000.0 / 1000.0))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file to use")
    return parser.parse_args()


def parse_yaml(yaml_file):
    global_config = yaml.load(open(yaml_file, 'rb'), Loader=yaml.Loader)
    yaml_dict = {}

    def add_dict(config, k, v):
        if not isinstance(v, dict):
            config[k] = v
            return
        for ik, iv in v.items():
            add_dict(config, ik, iv)

    add_dict(yaml_dict, "PreTraining", global_config["PreTraining"])
    args = argparse.Namespace(**yaml_dict)

    args.test_iters = args.eval_iters * 10

    # process batch size
    process_batch_size(args)

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size

    _print_args(args)
    model_size(args)
    return args, yaml_dict


def _print_args(args):
    """Print arguments."""
    print(
        '------------------------ arguments ------------------------',
        flush=True)
    str_list = []
    for arg in vars(args):
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print(
        '-------------------- end of arguments ---------------------',
        flush=True)
