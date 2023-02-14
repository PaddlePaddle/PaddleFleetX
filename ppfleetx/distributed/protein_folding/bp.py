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
""" Branch Parallel helper function"""

import paddle
from paddle.autograd import PyLayer
from . import scg

__all__ = [
    'get_world_size',
    'get_rank_in_group',
    ]

def get_world_size():
    nranks = 1
    if hasattr(scg, "bp_group"):
        nranks = scg.bp_group.nranks
    return nranks


def get_rank_in_group():
    rank = 0
    if hasattr(scg, "get_rank_in_bp_group"):
        rank = scg.get_rank_in_bp_group()
    return rank

@paddle.no_grad()
def broadcast(tensor, src):
    """ broadcast tensor from src rank in bp group """
    if get_world_size() == 1:
        return tensor
  
    assert src in [0, 1], "Branch Parallel is only support bp_degree=2 now!"
  
    group = scg.bp_group
    task = group.process_group.broadcast(tensor, src)
    task.wait()
    return tensor

class BroadcastGrad(PyLayer):
    """ A PyLayer Op broadcast gradient in backward stage """
    @staticmethod
    def forward(ctx, input, src):
        """ return input directly """ 
        ctx.src = src
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """ broadcast grad form src """ 
        broadcast(grad_output, ctx.src)
        return grad_output.clone()

def broadcast_grad_for_backward(input, src):
    """ a warpper for boradcast gradient in backward stage """
    if get_world_size() == 1:
        return input

    if not input.stop_gradient:
        output = BroadcastGrad.apply(input, src)
    else:
        output = input.clone()
    return output

@paddle.no_grad()
def all_reduce(tensor):
    """ allreduce a tensor in bp group """
    if get_world_size() == 1:
        return tensor

    group = scg.bp_group
    paddle.distributed.all_reduce(
        tensor, sync_op=True, group=group)

    return tensor



class SyncEvoformerResults(PyLayer):
    """ A PyLayer Op broadcast gradient in backward stage """
    @staticmethod
    def forward(ctx, outer, msa, pair):
        broadcast(outer, 0)
        if get_rank_in_group() == 1:
            pair += outer
        broadcast(pair, 1)
        broadcast(msa, 0)
        return msa, pair

    @staticmethod
    def backward(ctx, *grad_output):
        msa_grad = grad_output[0]
        pair_grad = grad_output[1]

        if get_rank_in_group() == 0:
            pair_grad = paddle.zeros_like(pair_grad)

        outer_grad = pair_grad.clone()
        broadcast(outer_grad, 1)
        
        return outer_grad, msa_grad, pair_grad

def sync_evoformer_results(outer, msa, pair):
    """ a warpper for boradcast gradient in backward stage """
    if get_world_size() == 1:
        return msa, pair

    if outer.stop_gradient and msa.stop_gradient and pair.stop_gradient:
        return msa, pair

    msa, pair = SyncEvoformerResults.apply(outer, msa, pair)
        
    return msa, pair

@paddle.no_grad()
def grad_sync(param_groups):
    """
        sync the gradients of params
    """

    nranks = get_world_size()

    if nranks < 2:
        return

    comm_group = scg.bp_group

    for group in param_groups:
        if group.get("bp", False):
            for p in group['params']:
                if p.is_distributed:
                    continue

                grad = p.grad
                if grad is None:
                    continue

                paddle.distributed.all_reduce(
                    grad, sync_op=True, group=comm_group)

    return None
