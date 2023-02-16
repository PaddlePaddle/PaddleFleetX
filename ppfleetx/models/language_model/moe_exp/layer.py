# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# The file has been adapted from a deepspeed file:
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/layer.py
# Git commit hash: a091bc223c01e94448f443456a6c15684644b966
# We retain the following license from the original files:
#   Copyright (c) The Microsoft DeepSpeed Team. All rights reserved.
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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .experts import Experts
from .sharded_moe import TopKGate, MOELayer


class MoE(nn.Layer):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 noisy_gate_policy=None,
                 drop_tokens=True,
                 use_rts=False,
                 enable_expert_tensor_parallelism=False):
        super(MoE, self).__init__()

        self.use_residual = use_residual
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        assert num_experts % ep_size == 0, f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size

        # log_dist(
        #     f'Creating MoE layer with num_experts: {num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size}',
        #     [0])

        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        experts = Experts(expert, self.num_local_experts,
                          self.expert_group_name)
        self.gate = TopKGate(hidden_size, num_experts, k, capacity_factor,
                             eval_capacity_factor, min_capacity,
                             noisy_gate_policy, drop_tokens, use_rts)
        self.fleetx_moe = MOELayer(self.gate, experts, self.expert_group_name,
                                   self.ep_size, self.num_local_experts)
        if self.use_residual:
            self.mlp = expert
            # coefficient is used for weighted sum of the output of expert and mlp
            self.coefficient = nn.Linear(hidden_size, 2)

    def forward(self, hidden_states, used_token=None):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.fleetx_moe(hidden_states, used_token)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(hidden_states)
            if type(output_mlp) is tuple:
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = F.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output
