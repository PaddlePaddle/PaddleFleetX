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

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.fluid import core
import argparse
from functools import reduce
from paddle import LazyGuard

sys.path.append("../../../../../")
from ppfleetx.core.module.basic_module import BasicModule
from ppfleetx.models.language_model.utils import process_global_configs, process_engine_configs
import ppfleetx.optims.lr_scheduler as lr


def parse_yaml_auto(yaml_dict):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--config", help="configuration file to use")
    # parser.add_argument(
    #     '-o',
    #     '--override',
    #     action='append',
    #     default=[],
    #     help='config options to be overridden')

    # yaml_args = parser.parse_args()

    # yaml_dict = GPTAutoConfig(
    #     yaml.load(
    #         open(yaml_args.config, 'rb'), Loader=yaml.Loader))
    # override_config(yaml_dict, yaml_args.override)

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
    process_global_configs(yaml_dict)
    # dp_degree = yaml_dict['Distributed']['dp_degree']
    # sharding_degree = yaml_dict['Distributed']['sharding']['sharding_degree']
    # data_configs = yaml_dict['Global']
    # if data_configs['global_batch_size'] is None:
    #     raise ValueError("global_batch_size should be set.")
    # elif data_configs['global_batch_size'] is not None:
    #     assert data_configs['global_batch_size'] % dp_degree == 0, \
    #         "global_batch_size[{}] should be divided by dp_degree[{}].".format(data_configs['global_batch_size'], dp_degree)
    #     assert dp_degree % sharding_degree == 0, \
    #         "dp_degree[{}] should be divided by sharding_degree[{}].".format(dp_degree, sharding_degree)

    # process model configs
    model_configs = yaml_dict['Model']
    if model_configs['ffn_hidden_size'] is None:
        model_configs['ffn_hidden_size'] = 4 * model_configs['hidden_size']

    # process engine configs
    process_engine_configs(yaml_dict)
    # engine_configs = yaml_dict['Engine']
    # engine_configs['test_iters'] = engine_configs[
    #     'eval_iters'] * 10 if engine_configs[
    #         'test_iters'] is None else engine_configs['test_iters']

    # _print_args(yaml_dict)
    # model_size(yaml_dict)
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


class GPTAutoModule(BasicModule):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs
        self.nranks = paddle.distributed.get_world_size()

        # from examples.gpt.tools import Mesh
        from .modeling_auto import GPTModel, GPTForPretraining, GPTPretrainingCriterion
        configs['Model']['mesh'] = Mesh(configs)

        with LazyGuard():
            self.model = GPTForPretraining(GPTModel(configs['Model']))
            self.loss_fn = GPTPretrainingCriterion(configs['Model']['mesh'])

        del configs['Model']['mesh']
        print('>> total parameters: ', len(self.model.parameters()))

    def get_model(self):
        pass

    def forward(self, tokens, ids):
        tokens.stop_gradient = True
        ids.stop_gradient = True

        return self.model(tokens, ids)

    def configure_optimizers(self):

        opt_configs = self.configs['Optimizer']
        warmup_step = opt_configs['lr']['warmup_rate'] * opt_configs['lr'][
            'decay_steps']
        lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
            max_lr=opt_configs['lr']['max_lr'],
            min_lr=opt_configs['lr']['min_lr'],
            warmup_rate=opt_configs['lr']['warmup_rate'],
            decay_steps=opt_configs['lr']['decay_steps'])

        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=opt_configs[
            'grad_clip']) if opt_configs['grad_clip'] > 0 else None

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler
            if lr_scheduler is not None else opt_configs['lr']['max_lr'],
            beta1=opt_configs['adam_beta1'],
            beta2=opt_configs['adam_beta2'],
            epsilon=opt_configs['adam_epsilon'],
            parameters=self.model.parameters(),
            weight_decay=opt_configs['weight_decay'],
            grad_clip=clip,
            apply_decay_param_fun=lambda x: x in decay_params)

        return optimizer
