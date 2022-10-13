#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
Distributed Data Parallel helper functions
"""

import paddle
from . import scg

__all__ = [
    'get_world_size',
    'get_rank_in_group',
    'grad_sync',
    'param_sync'
    ]

def get_world_size():
    nranks = 1
    if hasattr(scg, "dp_group"):
        nranks = scg.dp_group.nranks
    return nranks


def get_rank_in_group():
    rank = 0
    if hasattr(scg, "get_rank_in_dp_group"):
        rank = scg.get_rank_in_dp_group()
    return rank

@paddle.no_grad()
def grad_sync(param_groups, grad_avg=True):
    """
        sync the gradients of params
    """
    
    nranks = get_world_size()

    if nranks < 2:
        return

    comm_group = scg.dp_group

    for group in param_groups:
        for p in group['params']:
            if p.is_distributed:
                continue

            grad = p.grad
            if grad is None:
                continue

            paddle.distributed.all_reduce(
                grad, sync_op=True, group=comm_group)
            if grad_avg:
                grad = p.grad.scale_(1.0 / nranks)

    return None


@paddle.no_grad()
def param_sync(model, src_rank=0, comm_group=None):
    """
        broadcast params to other ranks
    """

    nranks = paddle.distributed.get_world_size(
    ) if comm_group is None else comm_group.nranks

    if nranks < 2:
        return

    for _, param in model._obtain_parameters_buffers().items():

        if param.is_distributed:
            continue

        if getattr(param, "no_sync", False):
            continue

        paddle.distributed.broadcast(
            param, src=src_rank, group=comm_group, sync_op=True)

    return None


@paddle.no_grad()
def all_reduce(tensor, op=paddle.distributed.ReduceOp.SUM):
    """ allreduce a tensor in bp group """
    if get_world_size() == 1:
        return tensor

    group = scg.dp_group
    paddle.distributed.all_reduce(
        tensor, sync_op=True, op=op, group=group)

    return tensor
