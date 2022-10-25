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
#     https://github.com/laekov/fastmoe/blob/master/fmoe/functions.py
#     Git commit hash: 295a615aacce7e54a37e7935274ba15e901c78e4
# We retain the following license from the original files:
#     Copyright 2021, Jiaao He. All rights reserved.
#   Licensed under the Apache License, Version 2.0 (the "License").

import paddle
from paddle.autograd import PyLayer
from paddle.distributed.utils.moe_utils import global_scatter, global_gather
from .utils import _local_scatter, _local_gather, _all_gather


class MoEScatter(PyLayer):
    r"""
    Scatter input samples from [batch x sequences] to contiguous alone experts.
    If `world_size` is greater than 1, the samples will first be locally
    scattered, and then exchanged across workers.
    """

    @staticmethod
    def forward(ctx,
                inp,
                pos,
                local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
                group=None):
        local_input_buf = _local_scatter(inp, pos)
        if world_size > 1:
            global_input_buf = global_scatter(
                local_input_buf,
                local_expert_count,
                global_expert_count,
                group=group)
        else:
            global_input_buf = local_input_buf

        ctx.moe_args = inp.shape[0], world_size, group

        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return global_input_buf

    @staticmethod
    def backward(ctx, grad):
        (pos, local_expert_count, global_expert_count) = ctx.saved_tensor()
        (inp_batch_size, world_size, group) = ctx.moe_args

        if world_size > 1:
            local_grad_in = global_gather(
                grad, local_expert_count, global_expert_count, group=group)
        else:
            local_grad_in = grad
        grad_in = _local_gather(local_grad_in, pos, inp_batch_size)
        return grad_in, None, None, None


class MoEGather(PyLayer):
    r"""
    Gather output samples from contiguous alone experts back to [batch x
    sequences]. Works symmetrically with MoEScatter.
    """

    @staticmethod
    def forward(ctx,
                global_output_buf,
                pos,
                local_expert_count,
                global_expert_count,
                local_batch_size,
                world_size,
                group=None):
        if world_size > 1:
            local_output_buf = global_gather(
                global_output_buf,
                local_expert_count,
                global_expert_count,
                group=group)
        else:
            local_output_buf = global_output_buf
        output = _local_gather(
            local_output_buf, pos, local_batch_size, maybe_overlap=False)

        ctx.moe_args = (global_output_buf.shape[0], world_size, group)
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        pos, local_expert_count, global_expert_count = ctx.saved_tensor()
        fwd_batch_size, world_size, group = ctx.moe_args
        grad_out_buf = _local_scatter(grad_out, pos)
        if world_size > 1:
            global_grad_out_buf = global_scatter(
                grad_out_buf,
                local_expert_count,
                global_expert_count,
                group=group)
        else:
            global_grad_out_buf = grad_out_buf
        return global_grad_out_buf, None, None, None


class AllGather(PyLayer):
    r"""
    A wrapper for the All-Gather function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        tensor_list = []
        paddle.distributed.all_gather(tensor_list, inp, group=group)
        output = paddle.concat(tensor_list, axis=0)
        ctx.args = rank, inp.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_out):
        rank, dim0 = ctx.args
        return paddle.slice(
            grad_out, axes=[0], starts=[rank * dim0], ends=[(rank + 1) * dim0])


class Slice(PyLayer):
    r"""
    A wrapper for the Slice function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        B = inp.shape[0]
        local_batch_size = B // world_size
        batch_start = local_batch_size * rank
        batch_end = min(batch_start + local_batch_size, B)
        inp = paddle.slice(
            inp, axes=[0], starts=[batch_start], ends=[batch_end])
        ctx.args = world_size, group
        return inp

    @staticmethod
    def backward(ctx, grad_out):
        world_size, group = ctx.args
        return _all_gather(grad_out, group=group)
