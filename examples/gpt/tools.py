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

import logging
import os
import sys

import yaml
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.fluid import core
import argparse
from functools import reduce
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


def is_fused_matmul_bias_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
        return hasattr(core.ops, 'fused_gemm_epilogue')
    else:
        return False


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

    if args.fused_linear and not is_fused_matmul_bias_supported():
        args.fused_linear = False
        logging.warning(
            "The flag fused_linear only valid for cuda version higher than 11.6, "
            "but the paddle is compiled with cuda " + paddle.version.cuda())

    if args.use_recompute:
        assert args.recompute_granularity is None or \
               isinstance(args.recompute_granularity, str), \
            "recompute_granularity must be a None or a string object"
        if args.recompute_granularity is None:
            args.recompute_granularity = "full"
        else:
            assert args.recompute_granularity in ["full", "only_attn"], \
                "recompute_granularity can be only chosen from " \
                "full or only_attn, but received " + args.recompute_granularity

    # process batch size
    process_batch_size(args)
    # process process_mesh
    process_mesh(args)

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


def process_mesh(args):
    args.mesh = Mesh(args)


class Mesh:
    def __init__(self, args):
        self.dp_idx = -1
        self.mp_idx = -1
        self.process_mesh = None
        self.args = args

        topology = list(
            filter(lambda x: x > 1,
                   [args.dp_degree, args.mp_degree, args.pp_degree]))
        num_proc = 1 if not topology else reduce(lambda x, y: x * y, topology)
        processes = [i for i in range(num_proc)]

        if args.pp_degree > 1:
            if len(topology) > 1:
                # dpmppp, dppp, mppp
                process_mesh_shape = topology[:-1]
                per_process_mesh_group = num_proc // args.pp_degree
                self.process_mesh = [
                    np.array(processes[i * per_process_mesh_group:(i + 1) *
                                       per_process_mesh_group]).reshape(
                                           process_mesh_shape).tolist()
                    for i in range(args.pp_degree)
                ]
                if len(process_mesh_shape) > 1:
                    self.dp_idx = 0
                    self.mp_idx = 1
                else:
                    self.dp_idx = 0 if args.dp_degree > 1 else -1
                    self.mp_idx = 0 if args.mp_degree > 1 else -1
            elif len(topology) == 1:
                # pp
                self.process_mesh = [[i] for i in range(args.pp_degree)]
        else:
            if len(topology) > 1:
                # dpmp
                self.process_mesh = [
                    np.array(processes).reshape(topology).tolist()
                ]
                self.dp_idx = 0
                self.mp_idx = 1
            else:
                # dp, mp, serial
                self.process_mesh = [processes]
                self.dp_idx = 0 if args.dp_degree > 1 else -1
                self.mp_idx = 0 if args.mp_degree > 1 else -1

    def __getitem__(self, idx):
        if self.args.pp_degree > 1:
            assert self.args.pp_degree == len(self.process_mesh)

        assert idx < len(self.process_mesh)
        return self.process_mesh[idx]

    def stages(self, num_layers):
        layer_per_stage = num_layers // self.args.pp_degree
        return [i // layer_per_stage for i in range(num_layers)]

    @property
    def dp(self):
        return self.dp_idx

    @property
    def mp(self):
        return self.mp_idx
