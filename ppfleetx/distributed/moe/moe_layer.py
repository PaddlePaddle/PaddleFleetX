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
#
# The file has been adapted from the file:
#     https://github.com/laekov/fastmoe/blob/master/fmoe/layers.py
#     Git commit hash: 295a615aacce7e54a37e7935274ba15e901c78e4
# We retain the following license from the original files:
#     Copyright 2021, Jiaao He. All rights reserved.
#   Licensed under the Apache License, Version 2.0 (the "License").

import numpy as np
import paddle
import paddle.nn as nn

from .gate import NaiveGate, GShardGate, SwitchGate, BaseGate
from .comm_ops import MoEScatter, MoEGather, AllGather, Slice
from .utils import prepare_forward
from paddle.incubate.distributed.fleet import recompute_hybrid


class MoELayer(nn.Layer):
    """MoE Layer
    Args:
        d_model: (int) model dimention
        experts: (nn.LayerList) expert networks list
        gate: (dict|NaiveGate|SwitchGate|NaiveGate):
                if gate is a dict:
                    gate is a gate network config, containing 2 keys:
                    `type`(str) value can be: "naive", "gshard", "switch" or None, default is "gshard"
                    `top_k`(int) default value is 2
                else gate is an instance of NaiveGate|SwitchGate|NaiveGate:

        moe_group: moe group for experts communication
        mp_group: mp group for mp commutication
        recompute_interval(int, optional): whether to use recompute, default 0, means to disable recompute.
        recompute_ctx(dict, optional): the context for recompute, if recompute_interval > 1, recompute_ctx must be given.
    Examples:
        .. code-block:: python
        from paddle.nn import layer, LayerList
        from paddle.distributed.moe import MoElayer
        from paddle.distributed.collective import Group
        from paddle.distributed import fleet

        moe_group = Group(fleet.worker_index(),
                          0,
                          list(range(fleet.worker_num())))
        mp_group = None

        num_experts=8
        dim_feedforward=512
        d_model=8
        top_k=2

        class ExpertLayer(Layer):
            def __init__(self, d_model, d_hidden, name=None,rank=0, windex = 0, num_expert=1):
                super(ExpertLayer, self).__init__()
                self.htoh4 = nn.Linear(d_model, d_hidden)
                self.h4toh = nn.Linear(d_hidden, d_model)

            def forward(self, x):
                x = self.htoh4(x)
                x = self.h4toh(x)
                return x

        gate_config = {
                "type": "gshard",
                "top_k": top_k,
        }

        experts_list = LayerList()
        for expi in range(num_experts):
            exp_layer = ExpertLayer(d_model, dim_feedforward // top_k, windex=expi, num_expert=num_experts)
            experts_list.append(exp_layer)

        moeLayer = MoELayer(d_model = d_model,
                            experts=experts_list,
                            gate=gate_config,
                            moe_group=moe_group,
                            mp_group=mp_group,
                            recompute_interval=0)

    """

    def __init__(self,
                 d_model,
                 experts,
                 gate=None,
                 moe_group=None,
                 mp_group=None,
                 recompute_interval=0,
                 recompute_ctx=None):
        super(MoELayer, self).__init__()

        self.recompute_ctx = recompute_ctx

        if gate is None:
            gate = dict()

        assert isinstance(gate, (dict, BaseGate)), \
             "gate config' type must be dict or an instance of BaseGate"
        # only support mp/dp
        self.group = moe_group

        self.world_size = 1
        if self.group is not None:
            self.world_size = self.group.nranks
        self.num_expert = len(experts)
        self.recompute_interval = recompute_interval
        assert experts is not None
        self.experts = experts

        self.mp_group = mp_group
        self.d_model = d_model
        if isinstance(gate, dict):
            self.top_k = gate.get("top_k", 2)
            gate = gate.get("type", "gshard")
            if gate == "naive" or gate is None:
                gate = NaiveGate(
                    self.d_model,
                    num_expert=len(experts),
                    world_size=self.world_size,
                    topk=self.top_k)
            elif gate == "gshard":
                gate = GShardGate(
                    self.d_model,
                    num_expert=len(experts),
                    world_size=self.world_size,
                    topk=self.top_k,
                    group=self.group)
            elif gate == "switch":
                gate = SwitchGate(
                    self.d_model,
                    num_expert=len(experts),
                    world_size=self.world_size,
                    topk=self.top_k,
                    group=self.group)
            else:
                assert False, "We only support naive gate, \
                                gshard gate and switch gate, \
                                but you choose {} gate.".format(str(gate))
        elif isinstance(gate, NaiveGate):
            self.top_k = gate.top_k
        elif isinstance(gate, BaseGate):
            raise TypeError("Unimplemented gate type: ", type(gate))
        else:
            raise TypeError("gate's type must be either dict or moe.BaseGate")
        self.gate = gate

    def forward(self, inp):
        origin_shape = inp.shape
        inp = inp.reshape_([-1, origin_shape[-1]])

        mp_rank = 0
        mp_size = 1
        if self.mp_group is not None:
            mp_rank = self.mp_group.rank
            mp_size = self.mp_group.nranks
        if mp_size > 1:
            inp = Slice.apply(inp, mp_rank, mp_size, self.mp_group)
        value, gate = self.gate(inp)

        (
            pos,
            local_expert_count,
            global_expert_count,
            fwd_expert_count,
            fwd_batch_size, ) = prepare_forward(gate, self.num_expert,
                                                self.world_size, self.group)

        topk = 1
        if len(gate.shape) == 2:
            topk = gate.shape[1]

        if pos.shape != [0]:
            temp_pos = pos // topk
        else:
            temp_pos = pos
        assert topk == self.top_k

        x = MoEScatter.apply(inp, temp_pos, local_expert_count,
                             global_expert_count, fwd_batch_size,
                             self.world_size, self.group)

        d_model = self.d_model

        def experts_fwd(x, fwd_expert_count, experts):

            if x.shape[0] == 0:
                return x
            y = []
            last_index = 0
            assert isinstance(fwd_expert_count, np.ndarray)
            assert len(experts) == len(fwd_expert_count)
            for idx, expert_count in enumerate(fwd_expert_count):
                if expert_count <= 0:
                    continue
                y.append(experts[idx](x[last_index:expert_count + last_index]))
                last_index = expert_count + last_index
            return paddle.concat(y, axis=0)

        if self.recompute_interval <= 0 or x.shape[0] == 0:
            x = experts_fwd(x, fwd_expert_count.numpy(), self.experts)
        else:
            x = recompute_hybrid(self.recompute_ctx, experts_fwd, x,
                                 fwd_expert_count.numpy(), self.experts)

        out_batch_size = inp.shape[0]
        if len(gate.shape) == 2:
            out_batch_size *= gate.shape[1]

        x = MoEGather.apply(x, pos, local_expert_count, global_expert_count,
                            out_batch_size, self.world_size, self.group)

        x = x.reshape([-1, self.top_k, d_model])
        value = value.reshape([x.shape[0], 1, self.top_k])
        x = paddle.bmm(value, x).reshape([-1, d_model])

        if mp_size > 1:
            x = AllGather.apply(x, mp_rank, mp_size, self.mp_group)

        x = paddle.reshape_(x, origin_shape)

        return x
