# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# The file has been adapted from a deepspeed file:
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/experts.py
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

import paddle
import paddle.nn as nn
import copy


class Experts(nn.Layer):
    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(Experts, self).__init__()

        self.fleetx_experts = nn.LayerList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.fleetx_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs):
        chunks = paddle.chunk(inputs, chunks=self.num_local_experts, axis=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.fleetx_experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_output = paddle.concat(expert_outputs, axis=1)
        return expert_output
