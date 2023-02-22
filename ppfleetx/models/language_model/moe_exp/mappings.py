# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# The file has been adapted from a deepspeed file:
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/mappings.py
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
import paddle.distributed as dist
from paddle.autograd import PyLayer


#TODO: set axis for all_gather
def _gather_tokens(input_, group, axis=0):
    """Gather tensors and concatenate them along a axisension"""
    # in case model is not deployed in distributed environment
    group = dist.collective._get_default_group() if group is None else group
    tensor_list = [paddle.empty_like(input_) for _ in range(group.nranks)]
    dist.all_gather(tensor_list, input_, group)
    output_ = paddle.concat(tensor_list, axis=axis)
    return output_


def _drop_tokens(input_, group, axis=0):
    """Divide a tensor among the tensor parallel ranks"""
    # in case model is not deployed in distributed environment
    group = dist.collective._get_default_group() if group is None else group

    total_chunks = group.nranks
    this_chunk = group.rank
    assert input_.shape[
        axis] % total_chunks == 0, f"input dimention {axis} ({input_.shape[axis]}) is not divisible by tensor parallel world size ({total_chunks})"
    chunk_size = input_.shape[axis] // total_chunks

    return paddle.slice(input_, [axis], [this_chunk * chunk_size],
                        [this_chunk * chunk_size + chunk_size])


class _GatherTokens(PyLayer):
    """All gather tokens among the tensor parallel ranks"""

    @staticmethod
    def forward(ctx, input_, group, axis):
        ctx.group = group
        ctx.axis = axis
        return _gather_tokens(input_, group, axis)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_tokens(grad_output, ctx.group, ctx.axis), None


class _DropTokens(PyLayer):
    "Divide tokens equally among the tensor parallel ranks"

    @staticmethod
    def forward(ctx, input_, group, axis):
        ctx.group = group
        ctx.axis = axis
        return _drop_tokens(input_, axis)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_tokens(grad_output, ctx.group, ctx.axis), None


def gather_tokens(input_, group=None, axis=0):
    if group is None or group.nranks == 1:
        # no tensor parallelism for non-experts
        return input_
    return _GatherTokens.apply(input_, group, axis)


def drop_tokens(input_, group=None, axis=0):
    if group is None or group.nranks == 1:
        # no tensor parallelism for non-experts
        return input_
    return _DropTokens.apply(input_, group, axis)
