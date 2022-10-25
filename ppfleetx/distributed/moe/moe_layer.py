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
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.distributed.fleet import recompute_hybrid


class MoELayer(nn.Layer):
    """MoE Layer
    Args:
        d_model: (int) model dimention
        experts: (list|nn.LayerList) expert networks list
        gate: (str|BaseGate|None):
                if gate is a str, it can only be "naive", "gshard", "switch" or None, default is "naive"
                else gate is an instance of BaseGate
        
        top_k: (int) default value is 2
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
            def __init__(self, d_model, d_hidden, name=None):
                super(ExpertLayer, self).__init__()
                self.htoh4 = nn.Linear(d_model, d_hidden)
                self.h4toh = nn.Linear(d_hidden, d_model)

            def forward(self, x):
                x = self.htoh4(x)
                x = self.h4toh(x)
                return x

        experts_list = LayerList()
        for expi in range(num_experts):
            exp_layer = ExpertLayer(d_model, dim_feedforward)
            experts_list.append(exp_layer)

        moeLayer = MoELayer(d_model = d_model,
                            experts=experts_list,
                            gate="gshard",
                            top_k=2,
                            moe_group=moe_group,
                            mp_group=mp_group,
                            recompute_interval=0)

    """

    def __init__(self,
                 d_model,
                 experts,
                 moe_group=None,
                 mp_group=None,
                 top_k=2,
                 gate=None,
                 recompute_interval=0,
                 recompute_partition=False,
                 recompute_offload=False):
        super(MoELayer, self).__init__()

        self.d_model = d_model

        assert experts is not None
        assert isinstance(experts, (list, nn.LayerList)), \
             "The type of experts must be list or nn.LayerList"

        for i, exp in enumerate(experts):
            assert isinstance(
                exp,
                nn.Layer), "The type of experts[{}] must be nn.Layer".format(i)

        self.experts = nn.LayerList(experts) if isinstance(experts,
                                                           list) else experts
        self.num_expert = len(experts)

        gate = "naive" if gate is None else gate
        assert isinstance(gate, (str, BaseGate)), \
             "The type of gate must be str or an instance of BaseGate"
        self.top_k = top_k

        # only support mp/dp
        self.group = moe_group
        self.mp_group = mp_group

        self.world_size = self.group.nranks \
            if self.group is not None else 1

        if isinstance(gate, str):
            gate_map = {
                "naive": NaiveGate,
                "gshard": GShardGate,
                "switch": SwitchGate,
            }

            if gate in gate_map.keys():
                self.gate = gate_map[gate](self.d_model,
                                           num_expert=self.num_expert,
                                           topk=self.top_k,
                                           group=self.group)
            else:
                assert False, "We only support naive gate, \
                                gshard gate and switch gate, \
                                but you choose {} gate.".format(gate)
        elif isinstance(gate, BaseGate):
            self.gate = gate
        else:
            raise TypeError("The type of gate must be either str in ('naive', \
                'gshard', 'switch') or an instance of moe.BaseGate")

        self.recompute_interval = recompute_interval
        self.recompute_ctx = {
            "mp_group": self.mp_group,
            "offload": recompute_offload,
            "partition": recompute_partition,
        }

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
        elif self.world_size > 1:
            x = recompute_hybrid(self.recompute_ctx, experts_fwd, x,
                                 fwd_expert_count.numpy(), self.experts)
        else:
            x = recompute(experts_fwd, x,
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
