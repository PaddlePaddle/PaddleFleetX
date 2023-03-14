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

from ppfleetx.distributed.apis import env


def process_global_configs(config):
    """
    process global configs for hybrid parallel
    """
    dp_degree = config['Distributed']['dp_degree']
    sharding_degree = config['Distributed']['sharding']['sharding_degree']

    configs = config['Global']
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
        return hasattr(core.eager.ops.legacy, 'fused_gemm_epilogue')
    else:
        return False


def process_fused_configs(config):
    """
    process fused configs for hybrid parallel
    """

    nranks = dist.get_world_size()
    dp_degree = config['Distributed']['dp_degree']
    configs = config['Fused']
    if configs['tensor_fusion']:
        assert nranks == dp_degree, "tensor_fusion only support single card train or data parallel train"


def process_inference_configs(config):
    """
    process fused configs for hybrid parallel
    """
    configs = config['Inference']

    if configs['model_dir'] is None:
        configs['model_dir'] = config['Engine']['save_load']['output_dir']

    if configs['mp_degree'] is None:
        configs['mp_degree'] = config['Distributed']['mp_degree']


def process_model_configs(config):
    """
    process model configs for hybrid parallel
    """
    configs = config['Model']

    if configs['use_recompute']:
        if not configs['recompute_granularity']:
            configs['recompute_granularity'] = 'full'

    if configs['fused_linear'] and not is_fused_matmul_bias_supported():
        configs['fused_linear'] = False
        logging.warning(
            "The flag fused_linear only valid for cuda version higher than 11.6, "
            "but the paddle is compiled with cuda " + paddle.version.cuda())


def process_optim_configs(config):
    """
    process optim configs for hybrid parallel
    """
    config['Optimizer']['multi_precision'] = config['Engine']['mix_precision'][
        'use_pure_fp16']


def process_engine_configs(config):
    """
    process engine configs for hybrid parallel
    """
    configs = config['Engine']
    configs['test_iters'] = configs['eval_iters'] * 10 \
        if configs.get('test_iters', None) is None \
        else configs['test_iters']
    configs['accumulate_steps'] = config['Global']['local_batch_size'] \
        // config['Global']['micro_batch_size']


def process_configs(config):

    process_fused_configs(config)
    process_model_configs(config)
    process_optim_configs(config)
    process_inference_configs(config)

    return config
