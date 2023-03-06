# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# The file has been adapted from a deepspeed file:
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py
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
from typing import Callable, Dict, Tuple, Optional, Any
from paddle.distribution import Uniform, Gumbel
import paddle.nn.functional as F
from paddle import Tensor
import paddle.nn as nn
import paddle.distributed as dist
from paddle.autograd import PyLayer
import paddle.distributed.fleet as fleet

from .mappings import drop_tokens, gather_tokens

uniform_map: Dict[str, Callable] = {}
gumbel_map: Dict[str, Callable] = {}
exp_selection_uniform_map: Dict[str, Callable] = {}


def multiplicative_jitter(x, epsilon=1e-2):
    if epsilon == 0:
        return x
    device = paddle.get_device()
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = Uniform(
            low=paddle.to_tensor(1.0 - epsilon),
            high=paddle.to_tensor(1.0 + epsilon)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape):
    device = paddle.get_device()
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = paddle.to_tensor(1.0)
        zero = paddle.to_tensor(0.0)
        gumbel = Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

class _AllToAll(PyLayer):
    @staticmethod
    def forward(ctx: Any, group: dist.collective.Group,
                input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        output = paddle.empty_like(input)
        dist.alltoall_single(input, output, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return paddle.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape((a.shape[0], -1)) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return paddle.bmm(paddle.unsqueeze(a, 1),
                          paddle.unsqueeze(b, 2)).reshape((-1))
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return paddle.matmul(a.reshape((s, -1)).t(), b).reshape((e, c, m))
    elif rule == 'sec,ecm->sm':
        return paddle.matmul(
            a.reshape((a.shape[0], -1)), b.reshape((-1, b.shape[-1])))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape((k, -1)).t().reshape((s, m, k))
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return paddle.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return paddle.einsum(rule, a, b)

def _capacity(gates, capacity_factor, min_capacity):
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    capacity = paddle.ceil(
        (num_tokens / num_experts) * capacity_factor).astype(paddle.int64)
    if capacity < min_capacity:
        capacity = min_capacity.astype(paddle.int64)
    return capacity


def _top_idx(source, k):
    return paddle.topk(source, k=k, axis=0)[1]


def top1gating(logits,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor=None,
               noisy_gate_policy: Optional[str]=None,
               drop_tokens: bool=True,
               use_rts: bool=True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + \
            gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    gates = F.softmax(logits, axis=1)

    capacity = _capacity(gates,
                         paddle.to_tensor(capacity_factor),
                         paddle.to_tensor(min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = paddle.argmax(
        logits_w_noise if noisy_gate_policy == 'RSample' else gates, axis=1)
    num_experts = int(gates.shape[1])

    assert(0 <= indices1_s.min() and indices1_s.max() < num_experts)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = paddle.sum(mask1, axis=0).detach()

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = paddle.max(exp_counts)
        # dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX,
        #                 group=dist.get_world_group())
        # capacity = new_capacity
        group = dist.collective._get_default_group()
        task = group.process_group.all_reduce(new_capacity, dist.ReduceOp.MAX)
        task.wait()

    # Compute l_aux
    me = paddle.mean(gates, axis=0)
    ce = paddle.mean(mask1.astype("float32"), axis=0)
    l_aux = paddle.sum(me * ce) * num_experts

    # Random Token Selection
    if use_rts:
        device = paddle.get_device()
        uniform = exp_selection_uniform_map.get(device)
        if uniform is None:
            uniform = Uniform(
                low=paddle.to_tensor(0.0), high=paddle.to_tensor(1.0)).rsample
            exp_selection_uniform_map[device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[
        0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

    top_idx = _top_idx(mask1_rand, capacity)
    new_mask1 = paddle.zeros_like(mask1).put_along_axis_(
        indices=top_idx, values=1., axis=0)
    mask1 *= new_mask1

    # Compute locations in capacity buffer

    with paddle.amp.auto_cast(False, level='O2'):
        locations1 = paddle.cumsum(mask1.astype(paddle.float32), axis=0) - 1
        # Store the capacity location for each token
        locations1_s = paddle.sum(locations1 * mask1.astype(paddle.float32), axis=1)

    # Normalize gate probabilities
    mask1_float = mask1.astype("float32")
    gates = gates * mask1_float

    assert(0 <= locations1_s.astype(paddle.int32).min() and locations1_s.astype(paddle.int32).max() < capacity)
    locations1_sc = F.one_hot(locations1_s.astype(paddle.int32),
                              capacity).astype(paddle.float32)

    combine_weights = einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.astype("bool")

    return l_aux, combine_weights, dispatch_mask, exp_counts


def top2gating(logits: Tensor, capacity_factor: float,
               min_capacity: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, axis=1)

    capacity = _capacity(gates,
                         paddle.to_tensor(capacity_factor * 2),
                         paddle.to_tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = paddle.argmax(gates, axis=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape)
    # Replace top-expert with min value
    # logits_except1 = logits_w_noise.masked_fill(mask1.astype("bool"), float("-inf"))
    logits_except1 = paddle.where(
        mask1.astype("bool"),
        paddle.ones(logits_w_noise.shape) * float("-inf"), logits_w_noise)
    indices2_s = paddle.argmax(logits_except1, axis=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = paddle.cumsum(mask1, axis=0) - 1
    locations2 = paddle.cumsum(mask2, axis=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += paddle.sum(mask1, axis=0, keepdim=True)

    # gating decisions
    exp_counts = paddle.sum(mask1, axis=0).detach()

    # Compute l_aux
    me = paddle.mean(gates, axis=0)
    ce = paddle.mean(mask1.astype("float32"), axis=0)
    l_aux = paddle.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= paddle.less_than(locations1, capacity)
    mask2 *= paddle.less_than(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = paddle.sum(locations1 * mask1, axis=1)
    locations2_s = paddle.sum(locations2 * mask2, axis=1)

    # Normalize gate probabilities
    mask1_float = mask1.astype("float32")
    mask2_float = mask2.astype("float32")
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    # HACK: paddle currently does not support finfo, use constant instead
    min_constant = 1.1920928955078125e-07
    denom_s = paddle.clip(denom_s, min=min_constant)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = F.one_hot(locations1_s, capacity)
    locations2_sc = F.one_hot(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.astype("bool")

    return l_aux, combine_weights, dispatch_mask, exp_counts


class TopKGate(nn.Layer):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int=1,
                 capacity_factor: float=1.0,
                 eval_capacity_factor: float=1.0,
                 min_capacity: int=8,
                 noisy_gate_policy: Optional[str]=None,
                 drop_tokens: bool=True,
                 use_rts: bool=True) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.wg = nn.Linear(model_dim, num_experts).to(dtype=paddle.float32)
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        # self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts

    def forward(self, input: paddle.Tensor, used_token: paddle.Tensor=None
                ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        # if self.wall_clock_breakdown:
        #     self.timers('TopKGate').start()

        if self.wg.weight.dtype != paddle.float32:
            self.wg = self.wg.to(dtype=paddle.float32)
        input_fp32 = input.astype("float32")
        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32)
        logits = self.wg(input_fp32)

        if self.k == 1:
            gate_output = top1gating(
                logits, self.capacity_factor
                if self.training else self.eval_capacity_factor,
                self.min_capacity, used_token, self.noisy_gate_policy
                if self.training else None, self.drop_tokens, self.use_rts)

        else:
            gate_output = top2gating(logits, self.capacity_factor
                                     if self.training else
                                     self.eval_capacity_factor,
                                     self.min_capacity)

        # if self.wall_clock_breakdown:
        #     self.timers('TopKGate').stop()
        #     self.gate_time = self.timers('TopKGate').elapsed(reset=False)

        return gate_output


class MOELayer(nn.Layer):

    def __init__(self,
                 gate: nn.Layer,
                 experts: nn.Layer,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        # self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        #HACK need fix
        # self.hcg = fleet.get_hybrid_communicate_group()
        self.hcg = None

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def get_loss(self):
        return self.l_aux

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:

        # if self.wall_clock_breakdown:
        #     self.timers('moe').start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape((-1, d_model))

        self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(
            reshaped_input, input[1])
        dispatched_input = einsum("sec,sm->ecm",
                                  dispatch_mask.astype(input[0].dtype),
                                  reshaped_input)

        # if self.wall_clock_breakdown:
        #     self.timers('falltoall').start()

        # HACK: _get_expert_model_parallel_world_size is needed here
        if False and self.hcg.get_model_parallel_group().nranks == 1:
            # If the non-expert is tensor-parallel, it will create
            # duplicate tokens on the tensor-parallel ranks.
            # Since our experts are not tensor-parallel, these duplicates
            # need to be dropped to ensure correctness.
            # this also doubles up as a communication optimization as we are
            # reducing the all-to-all communication volume.
            dispatched_input = drop_tokens(dispatched_input, axis=1)

        # HACK disable AllToAll
        # dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        # if self.wall_clock_breakdown:
        #     self.timers('falltoall').stop()
        #     self.time_falltoall = self.timers('falltoall').elapsed(reset=False)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(
            (self.ep_size, self.num_local_experts, -1, d_model))

        expert_output = self.experts(dispatched_input)

        # if self.wall_clock_breakdown:
        #     self.timers('salltoall').start()

        # HACK disable AllToAll
        # expert_output = _AllToAll.apply(self.ep_group, expert_output)

        # if self.wall_clock_breakdown:
        #     self.timers('salltoall').stop()
        #     self.time_salltoall = self.timers('salltoall').elapsed(reset=False)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(
            (self.ep_size * self.num_local_experts, -1, d_model))

        # HACK: _get_expert_model_parallel_world_size is needed here
        if False and self.hcg.get_model_parallel_group().nranks == 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, axis=1)

        combined_output = einsum("sec,ecm->sm",
                                 combine_weights.astype(input[0].dtype),
                                 expert_output)

        a = combined_output.reshape((input[0].shape))

        # if self.wall_clock_breakdown:
        #     self.timers('moe').stop()
        #     self.time_moe = self.timers('moe').elapsed(reset=False)

        return a
