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
Dynamic Axial Parallelism and Duality Async Operation helper functions
paper ref: FastFold: Reducing AlphaFold Training Time from 11 Days to 67 Hours, https://arxiv.org/abs/2203.00854
code ref: https://github.com/hpcaitech/FastFold.git
"""

import warnings
import time
import paddle
from paddle import nn
from paddle import distributed as dist
from paddle.autograd import PyLayer
from . import scg

__all__ = [
    'set_dap_sync_op', 'get_dap_sync_op', 'get_world_size',
    'get_rank_in_group', 'scatter', 'gather', 'all_gather', 'all_gather_opp',
    'all_to_all', 'all_to_all_opp', 'row_to_col', 'col_to_row'
]

_sync_op = True


def set_dap_sync_op(sync_op):
    assert sync_op in [True, False]
    assert sync_op is True, "Only support sync mode now!"
    global _sync_op
    _sync_op = sync_op


def get_dap_sync_op():
    global _sync_op
    return _sync_op


def get_world_size():
    nranks = 1
    if hasattr(scg, "dap_group"):
        nranks = scg.dap_group.nranks
    return nranks


def get_rank_in_group():
    rank = 0
    if hasattr(scg, "get_rank_in_dap_group"):
        rank = scg.get_rank_in_dap_group()
    return rank


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


@paddle.no_grad()
def _all_gather(tensor, axis=-1, sync_op=True):
    group = scg.dap_group
    tensor_shape = list(tensor.shape)
    tensor_shape[0] *= group.nranks
    out = paddle.zeros(tensor_shape, tensor.dtype)
    out.stop_gradient = tensor.stop_gradient
    task = group.process_group.all_gather(tensor, out)
    task.wait()
    return out


@paddle.no_grad()
def _gather(tensor, axis=-1):
    output = _all_gather(tensor)
    if axis != 0:
        output = paddle.concat(
            paddle.split(
                output, get_world_size(), axis=0), axis=axis)
    return output


@paddle.no_grad()
def _split(tensor, axis=-1):
    ensure_divisibility(tensor.shape[axis], get_world_size())
    tensor_list = paddle.split(tensor, get_world_size(), axis=axis)

    output = tensor_list[get_rank_in_group()]

    return output


class Scatter(PyLayer):
    """ Scatter PyLayer Op"""

    @staticmethod
    def forward(ctx, input, axis: -1):
        ctx.axis = axis
        return _split(input, axis=axis)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, axis=ctx.axis)


def scatter(input, axis=-1):
    """ split a tensor according axis by dap size """
    if get_world_size() == 1:
        return input

    if not input.stop_gradient:
        output = Scatter.apply(input, axis=axis)
    else:
        output = _split(input, axis=axis)
    return output


class Gather(PyLayer):
    """ Gather PyLayer Op """

    @staticmethod
    def forward(ctx, input, axis=-1):
        ctx.axis = axis
        return _gather(input, axis=axis)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, axis=ctx.axis)


def gather(input, axis=-1):
    """ gather tensor form all rank in dap group in axis """
    if get_world_size() == 1:
        return input

    if not input.stop_gradient:
        output = Gather.apply(input, axis=axis)
    else:
        output = _gather(input, axis=axis)
    return output


@paddle.no_grad()
def _reduce_scatter(tensor, sync_op=True):
    group = scg.dap_group
    tensor_shape = list(tensor.shape)
    tensor_shape[0] = divide(tensor_shape[0], group.nranks)
    output = paddle.zeros(tensor_shape, tensor.dtype)
    output.stop_gradient = tensor.stop_gradient
    task = group.process_group._reduce_scatter_base(
        output, tensor, paddle.fluid.core.ReduceOp.SUM)
    task.wait()
    return output


class AllGather(PyLayer):
    """ AllGather PyLayer Op """

    @staticmethod
    def forward(ctx, input, axis=-1, sync_op=True):
        ctx.axis = axis
        ctx.sync_op = sync_op
        output = _all_gather(input, axis=axis, sync_op=sync_op)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.sync_op:
            pass
            # TODO(GuoxiaWang): implement wait logical
        return grad_output


class AllGather_Opp(PyLayer):
    """ Duality Async Operation for AllGather """

    @staticmethod
    def forward(ctx, input, axis=-1, sync_op=True):
        ctx.axis = axis
        ctx.sync_op = sync_op
        return input

    @staticmethod
    def backward(ctx, grad_output):
        output = _reduce_scatter(grad_output, sync_op=ctx.sync_op)
        return output


def all_gather(input, axis=-1):
    """ gather tensors from all rank in dap group and all get the result.
        if sync_op=None, sync will be assign according init_dap setting.

        when using async communication, sync_op=False, do not use the output as same as input.
        E.g. do not use `a = all_gather(a, ...)`, recommend to use `b = all_gather(a, ...)`
    """
    if get_world_size() == 1:
        return input

    sync_op = get_dap_sync_op()

    if not input.stop_gradient:
        output = AllGather.apply(input, axis, sync_op=sync_op)
    else:
        output = _all_gather(input, axis, sync_op=sync_op)
    return output


def all_gather_opp(output, axis=-1):
    """ Duality Async Operation for all_gather.
        if sync_op=None, sync will be assign according init_dap setting.
    """
    nranks = get_world_size()
    if nranks == 1:
        return output

    sync_op = get_dap_sync_op()

    if not sync_op:
        # TODO(GuoxiaWang): implement wait logical
        pass

    if not output.stop_gradient:
        output = AllGather_Opp.apply(output, axis, sync_op=sync_op)

    if axis != 0:
        output = paddle.concat(paddle.split(output, nranks, 0), axis=axis)

    return output


@paddle.no_grad()
def _all_to_all(tensor, in_axis=-1, out_axis=-1, sync_op=True):
    group = scg.dap_group
    tensor_shape = list(tensor.shape)

    out = paddle.zeros(tensor_shape, tensor.dtype)
    out.stop_gradient = tensor.stop_gradient
    task = group.process_group.alltoall(tensor, out)
    task.wait()

    return out


class All_to_All(PyLayer):
    """ All_to_All PyLayer Op"""

    @staticmethod
    def forward(ctx, input, in_axis=-1, out_axis=-1, sync_op=True):
        ctx.in_axis = in_axis
        ctx.out_axis = out_axis
        ctx.sync_op = sync_op
        return _all_to_all(
            input, in_axis=in_axis, out_axis=out_axis, sync_op=sync_op)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.sync_op:
            # TODO(GuoxiaWang): implement wait logical
            pass
        return grad_output


class All_to_All_Opp(PyLayer):
    """ Duality Async Operation for All_to_All """

    @staticmethod
    def forward(ctx, output, in_axis=-1, out_axis=-1, sync_op=True):
        ctx.in_axis = in_axis
        ctx.out_axis = out_axis
        ctx.sync_op = sync_op
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return _all_to_all(
            grad_output,
            in_axis=ctx.out_axis,
            out_axis=ctx.in_axis,
            sync_op=ctx.sync_op)


def all_to_all(input, in_axis, out_axis):
    """ all to all according in_axis and out_axis.
        if sync_op=None, sync will be assign according init_dap setting.
    """
    if get_world_size() == 1:
        return input

    sync_op = get_dap_sync_op()

    if in_axis != 0:
        ensure_divisibility(input.shape[in_axis], get_world_size())
        input = paddle.concat(
            paddle.split(
                input, get_world_size(), axis=in_axis), axis=0)

    if not input.stop_gradient:
        output = All_to_All.apply(
            input, in_axis=in_axis, out_axis=out_axis, sync_op=sync_op)
    else:
        output = _all_to_all(
            input, in_axis=in_axis, out_axis=out_axis, sync_op=sync_op)

    return output


def all_to_all_opp(output, in_axis, out_axis):
    """ Duality Async Operation for all_to_all.
        if sync_op=None, sync will be assign according init_dap setting.
    """
    if get_world_size() == 1:
        return output

    sync_op = get_dap_sync_op()

    if not sync_op:
        # TODO(GuoxiaWang): implement wait logical
        pass

    if not output.stop_gradient:
        output = All_to_All_Opp.apply(
            output, in_axis=in_axis, out_axis=out_axis, sync_op=sync_op)

    if out_axis != 0:
        ensure_divisibility(output.shape[0], get_world_size())
        output = paddle.concat(
            paddle.split(
                output, get_world_size(), axis=0), axis=out_axis)

    return output


class All2All(PyLayer):
    @staticmethod
    def forward(ctx, input, in_axis=-1, out_axis=-1):
        ctx.in_axis = in_axis
        ctx.out_axis = out_axis
        return _all_to_all(input, in_axis=in_axis, out_axis=out_axis)

    @staticmethod
    def backward(ctx, grad_output):
        return _all_to_all(
            grad_output, in_axis=ctx.out_axis, out_axis=ctx.in_axis)


def row_to_col(input):
    """ N, S, R, C => N, R, S, C using sync all_to_all """
    if get_world_size() == 1:
        return input

    ensure_divisibility(input.shape[2], get_world_size())
    input = paddle.concat(
        paddle.split(
            input, get_world_size(), axis=2), axis=0)

    if not input.stop_gradient:
        output = All2All.apply(input, in_axis=2, out_axis=1)
    else:
        output = _all_to_all(input, in_axis=2, out_axis=1)

    output = paddle.concat(
        paddle.split(
            output, get_world_size(), axis=0), axis=1)
    return output


def col_to_row(input):
    """ N, R, S, C => N, S, R, C using sync all_to_all """
    if get_world_size() == 1:
        return input

    ensure_divisibility(input.shape[1], get_world_size())
    input = paddle.concat(
        paddle.split(
            input, get_world_size(), axis=1), axis=0)

    if not input.stop_gradient:
        output = All2All.apply(input, in_axis=1, out_axis=2)
    else:
        output = _all_to_all(input, in_axis=1, out_axis=2)

    output = paddle.concat(
        paddle.split(
            output, get_world_size(), axis=0), axis=2)
    return output


@paddle.no_grad()
def grad_sync(param_groups):
    """
        sync the gradients of params
    """

    nranks = get_world_size()

    if nranks < 2:
        return

    comm_group = scg.dap_group

    for group in param_groups:
        if group.get("dap", False):
            for p in group['params']:
                if p.is_distributed:
                    continue

                grad = p.grad
                if grad is None:
                    continue

                paddle.distributed.all_reduce(
                    grad, sync_op=True, group=comm_group)

    return None
