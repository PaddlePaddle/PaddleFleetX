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
import copy

import yaml
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.fluid import core
import argparse
from functools import reduce
from fleetx.datasets.gpt import create_pretrained_dataset, get_train_data_file
from .gpt_config import GPTConfig, GPTAutoConfig
from fleetx.utils import logger


def process_dist_configs(yaml_dict):
    """
    process distributed strategy for hybrid parallel
    """
    configs = yaml_dict['Distributed']

    nranks = dist.get_world_size()
    other_degree = configs['mp_degree'] * configs['pp_degree'] * configs[
        'sharding']['sharding_degree']
    assert nranks % other_degree == 0, "unreasonable configs of dist_strategy."

    if configs['dp_degree'] * other_degree != nranks:
        logger.warning('Mismatched configs using {} cards with dp_degree[{}], ' \
            'mp_degree[{}], pp_degree[{}] and sharding_degree[{}]. So adaptively ' \
            'adjust dp_degree to {}'.format(nranks, configs['dp_degree'], configs['mp_degree'],
            configs['pp_degree'], configs['sharding']['sharding_degree'], nranks // other_degree))

    configs['dp_degree'] = nranks // other_degree
    assert nranks % configs[
        'dp_degree'] == 0, "unreasonable configs of dist_strategy."


def process_data_configs(yaml_dict):
    """
    process data configs for hybrid parallel
    """
    dp_degree = yaml_dict['Distributed']['dp_degree']
    sharding_degree = yaml_dict['Distributed']['sharding']['sharding_degree']

    configs = yaml_dict['Data']['batch_size']
    if configs['global_batch_size'] is None and configs[
            'local_batch_size'] is None:
        raise ValueError(
            "global_batch_size or local_batch_size should be set.")
    elif configs['global_batch_size'] is not None and configs[
            'local_batch_size'] is not None:
        assert configs['global_batch_size'] // configs['local_batch_size'] == (dp_degree * sharding_degree), "global_batch_size[{}] should be divided by local_batch_size[{}] "\
            "when dp_degree is [{}] and sharding_degree is [{}]".format(configs['global_batch_size'],
            configs['local_batch_size'], dp_degree, sharding_degree)
    elif configs['global_batch_size'] is not None and configs[
            'local_batch_size'] is None:
        assert configs['global_batch_size'] % (dp_degree * sharding_degree) == 0, \
            "global_batch_size[{}] should be divided by dp_degree[{}] times sharding_degree[{}]"\
            .format(configs['global_batch_size'], dp_degree, sharding_degree)
        configs['local_batch_size'] = configs['global_batch_size'] // (
            dp_degree * sharding_degree)
    else:
        configs['global_batch_size'] = configs[
            'local_batch_size'] * dp_degree * sharding_degree
    assert configs['local_batch_size'] % configs['micro_batch_size'] == 0


def is_fused_matmul_bias_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
        return hasattr(core.ops, 'fused_gemm_epilogue')
    else:
        return False


def process_fused_configs(yaml_dict):
    """
    process fused configs for hybrid parallel
    """
    nranks = dist.get_world_size()
    dp_degree = yaml_dict['Distributed']['dp_degree']

    configs = yaml_dict['Fused']
    if configs['tensor_fusion']:
        assert nranks == dp_degree, "tensor_fusion only support single card train or data parallel train"


def process_model_configs(yaml_dict):
    """
    process model configs for hybrid parallel
    """
    configs = yaml_dict['Model']
    if configs['ffn_hidden_size'] is None:
        configs['ffn_hidden_size'] = 4 * configs['hidden_size']

    if configs['use_recompute']:
        assert configs['recompute_granularity'] in ["full", "only_attn"], \
                "recompute_granularity can be only chosen from " \
                "'full' or 'only_attn', but received '{}'".format(configs['recompute_granularity'])

    if configs['fused_linear'] and not is_fused_matmul_bias_supported():
        configs['fused_linear'] = False
        logging.warning(
            "The flag fused_linear only valid for cuda version higher than 11.6, "
            "but the paddle is compiled with cuda " + paddle.version.cuda())


def process_engine_configs(yaml_dict):
    """
    process engine configs for hybrid parallel
    """
    configs = yaml_dict['Engine']
    configs['test_iters'] = configs['eval_iters'] * 10 if configs[
        'test_iters'] is None else configs['test_iters']
    configs['accumulate_steps'] = yaml_dict['Data']['batch_size']['local_batch_size'] \
        // yaml_dict['Data']['batch_size']['micro_batch_size']


def model_size(yaml_dict):
    """
    get model size for transformer
    """
    l = yaml_dict['Model']['num_layers']
    h = yaml_dict['Model']['hidden_size']
    V = yaml_dict['Model']['vocab_size']
    S = yaml_dict['Data']['dataset']['max_seq_len']
    P = 12 * l * h * h * (1 + 13 / (12 * h) + (V + S) / (12 * l * h))

    print('Model Size: {:.2f} B'.format(P / 1000.0 / 1000.0 / 1000.0))


def override(dl, ks, v):
    """
    Recursively replace dict of list
    """

    def str2num(v):
        try:
            return eval(v)
        except Exception:
            return v

    assert isinstance(dl, (list, dict)), ("{} should be a list or a dict")
    assert len(ks) > 0, ('lenght of keys should larger than 0')
    if isinstance(dl, list):
        k = str2num(ks[0])
        if len(ks) == 1:
            assert k < len(dl), ('index({}) out of range({})'.format(k, dl))
            dl[k] = str2num(v)
        else:
            override(dl[k], ks[1:], v)
    else:
        if len(ks) == 1:
            # assert ks[0] in dl, ('{} is not exist in {}'.format(ks[0], dl))
            if not ks[0] in dl:
                print('A new filed ({}) detected!'.format(ks[0], dl))
            dl[ks[0]] = str2num(v)
        else:
            override(dl[ks[0]], ks[1:], v)


def override_config(config, options=None):
    """
    Recursively override the config
    """
    if options is not None:
        for opt in options:
            assert isinstance(opt, str), (
                "option({}) should be a str".format(opt))
            assert "=" in opt, (
                "option({}) should contain a ="
                "to distinguish between key and value".format(opt))
            pair = opt.split('=')
            assert len(pair) == 2, ("there can be only a = in the option")
            key, value = pair
            keys = key.split('.')
            override(config, keys, value)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    return parser.parse_args()


def parse_yaml(yaml_args):
    yaml_dict = GPTConfig(
        yaml.load(
            open(yaml_args.config, 'rb'), Loader=yaml.Loader))
    override_config(yaml_dict, yaml_args.override)

    process_dist_configs(yaml_dict)
    process_data_configs(yaml_dict)
    process_fused_configs(yaml_dict)
    process_model_configs(yaml_dict)
    process_engine_configs(yaml_dict)

    _print_args(yaml_dict)
    model_size(yaml_dict)
    return yaml_dict


def _print_args(yaml_dict):
    """Print arguments."""
    args = {}

    def add_dict(config, k, v):
        if not isinstance(v, dict):
            config[k] = v
            return
        for ik, iv in v.items():
            add_dict(config, ik, iv)

    for key, value in yaml_dict.items():
        add_dict(args, key, value)

    print(
        '------------------------ arguments ------------------------',
        flush=True)
    str_list = []
    for key, value in args.items():
        dots = '.' * (48 - len(key))
        str_list.append('  {} {} {}'.format(key, dots, value))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print(
        '-------------------- end of arguments ---------------------',
        flush=True)


def parse_yaml_auto(yaml_args):
    yaml_dict = GPTAutoConfig(
        yaml.load(
            open(yaml_args.config, 'rb'), Loader=yaml.Loader))
    override_config(yaml_dict, yaml_args.override)

    # process dist configs
    dist_configs = yaml_dict['Distributed']
    nranks = dist.get_world_size()
    other_degree = dist_configs['mp_degree'] * dist_configs['pp_degree']
    assert nranks % other_degree == 0, "Requires nranks should be divided by mp_degree*pp_degree."
    if dist_configs['dp_degree'] * other_degree != nranks:
        logger.warning('Mismatched configs using {} cards with dp_degree[{}], ' \
            'mp_degree[{}], pp_degree[{}] and sharding_degree[{}]. So adaptively ' \
            'adjust dp_degree to {}'.format(nranks, dist_configs['dp_degree'], dist_configs['mp_degree'],
            dist_configs['pp_degree'], dist_configs['sharding']['sharding_degree'], nranks // other_degree))
    dist_configs['dp_degree'] = nranks // other_degree
    assert nranks % dist_configs[
        'dp_degree'] == 0, "unreasonable configs of dist_strategy."

    # process data configs
    dp_degree = yaml_dict['Distributed']['dp_degree']
    sharding_degree = yaml_dict['Distributed']['sharding']['sharding_degree']
    data_configs = yaml_dict['Data']['batch_size']
    if data_configs['global_batch_size'] is None:
        raise ValueError("global_batch_size should be set.")
    elif data_configs['global_batch_size'] is not None:
        assert data_configs['global_batch_size'] % dp_degree == 0, \
            "global_batch_size[{}] should be divided by dp_degree[{}].".format(data_configs['global_batch_size'], dp_degree)
        assert dp_degree % sharding_degree == 0, \
            "dp_degree[{}] should be divided by sharding_degree[{}].".format(dp_degree, sharding_degree)

    # process model configs
    model_configs = yaml_dict['Model']
    if model_configs['ffn_hidden_size'] is None:
        model_configs['ffn_hidden_size'] = 4 * model_configs['hidden_size']

    # process engine configs
    engine_configs = yaml_dict['Engine']
    engine_configs['test_iters'] = engine_configs[
        'eval_iters'] * 10 if engine_configs[
            'test_iters'] is None else engine_configs['test_iters']

    _print_args(yaml_dict)
    model_size(yaml_dict)
    return yaml_dict


class Mesh:
    def __init__(self, configs):
        self.dp_idx = -1
        self.mp_idx = -1
        self.process_mesh = None
        self.configs = configs['Distributed']

        topology = list(
            filter(lambda x: x > 1, [
                self.configs['dp_degree'], self.configs['mp_degree'],
                self.configs['pp_degree']
            ]))
        num_proc = 1 if not topology else reduce(lambda x, y: x * y, topology)
        processes = [i for i in range(num_proc)]

        if self.configs['pp_degree'] > 1:
            if len(topology) > 1:
                # dpmppp, dppp, mppp
                process_mesh_shape = topology[:-1]
                per_process_mesh_group = num_proc // self.configs['pp_degree']
                self.process_mesh = [
                    np.array(processes[i * per_process_mesh_group:(i + 1) *
                                       per_process_mesh_group]).reshape(
                                           process_mesh_shape).tolist()
                    for i in range(self.configs['pp_degree'])
                ]
                if len(process_mesh_shape) > 1:
                    self.dp_idx = 0
                    self.mp_idx = 1
                else:
                    self.dp_idx = 0 if self.configs['dp_degree'] > 1 else -1
                    self.mp_idx = 0 if self.configs['mp_degree'] > 1 else -1
            elif len(topology) == 1:
                # pp
                self.process_mesh = [[i]
                                     for i in range(self.configs['pp_degree'])]
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
                self.dp_idx = 0 if self.configs['dp_degree'] > 1 else -1
                self.mp_idx = 0 if self.configs['mp_degree'] > 1 else -1

    def __getitem__(self, idx):
        if self.configs['pp_degree'] > 1:
            assert self.configs['pp_degree'] == len(self.process_mesh)

        assert idx < len(self.process_mesh)
        return self.process_mesh[idx]

    def stages(self, num_layers):
        layer_per_stage = num_layers // self.configs['pp_degree']
        return [i // layer_per_stage for i in range(num_layers)]

    @property
    def dp(self):
        return self.dp_idx

    @property
    def mp(self):
        return self.mp_idx
