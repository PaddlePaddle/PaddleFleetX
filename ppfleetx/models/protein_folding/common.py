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

import numpy as np
import functools
import numbers
import collections
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute

try:
    from paddle import _legacy_C_ops as _C_ops
except:
    from paddle import _C_ops


def set_tensor_constant(tensor, constant):
    tensor.set_value(paddle.full_like(tensor, constant))


def init_gate_linear(linear):
    set_tensor_constant(linear.weight, 0)
    set_tensor_constant(linear.bias, 1)


def init_final_linear(linear):
    set_tensor_constant(linear.weight, 0)


def recompute_wrapper(func, *args, is_recompute=True):
    """Function wrapper for recompute"""
    if is_recompute:
        return recompute(func, *args)
    else:
        return func(*args)


def subbatch(f, arg_idx, dim, bs, out_idx, same_arg_idx={}):
    """ Converts a function to one that applies to subbatch of an input
    dimension.
    Args:
        f(Callable): original function.
        arg_idx([int]): indices of the inputs to be subbatched.
        dim([int]): index of the dimension to be subbatched.
        bs(int): subbatch size.
        out_idx(int): index of the output dimension that needs stacking
        same_arg_idx(dict), optional: index of same arg mapping. e.g {1: 0} means arg[1] == arg[0],
                            we assign _args[1] = _args[0] avoiding slice repeatly.
    Returns:
        converted function.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        assert len(arg_idx) == len(
            dim
        ), f'Number of batching args and number of batching dims should match.'

        inps = [args[i] for i in arg_idx]
        dim_width = [inp.shape[d] for inp, d in zip(inps, dim)]
        assert len(set(dim_width)) == 1, f'Batch sizes should be kept equal.'

        inp_dim = {inp: d for inp, d in zip(inps, dim)}

        dim_width = dim_width[0]
        if dim_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in np.arange(0, dim_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in same_arg_idx:
                    assert i > same_arg_idx[
                        i], f"expect i > same_arg_idx[i], but got i: {i} and same_arg_idx[i]: {same_arg_idx[i]}"
                    _args.append(_args[same_arg_idx[i]])
                elif i in arg_idx:
                    inp = inp.slice([inp_dim[inp]], [slice_at],
                                    [slice_at + bs])
                    _args.append(inp)
                else:
                    _args.append(inp)
            outs.append(f(*_args, **kwargs))

        return paddle.concat(outs, out_idx)

    return wrapper


def batched_gather(params, indices, axis=0, batch_dims=0):
    # Implement gather with batching, like tensorflow:
    # https://www.tensorflow.org/api_docs/python/tf/gather#batching
    # print(params.shape, indices.shape, axis)
    p, i = params, indices
    rank = len(p.shape)
    axis = (rank + axis) % rank
    # The stride of axis
    stride = p.shape[batch_dims + axis]

    if batch_dims == 0 and len(i.shape) == 1:
        return paddle.gather(p, i, axis=axis)

    elif batch_dims == 0:
        flat_i = i.reshape([-1])
        gathered = paddle.gather(p, flat_i, axis=axis)
        shape = p.shape[:axis] + i.shape
        if axis < rank - 1:
            shape += params.shape[axis + 1:]
        return gathered.reshape(shape)

    b = batch_dims
    a = axis
    assert p.shape[:b] == i.shape[:b]
    bn = np.prod(p.shape[:b])

    # Shift batch dimensions right to bundle with axis
    if a > 0:
        perm = list(range(rank))
        perm = perm[b:(b + a)] + perm[:b] + perm[(b + a):]
        p = p.transpose(perm)

    # Merge params' batch+axis
    p = p.reshape(p.shape[:a] + [-1] + p.shape[(b + a + 1):])

    # indices = [Batch..., Index...]
    # Expand the index values across batch elements
    strides = paddle.arange(bn).unsqueeze(-1) * stride
    i = i.reshape([bn, -1])
    flat_i = paddle.flatten(i + strides)

    # Do gather
    gathered = paddle.gather(p, flat_i, axis=axis)

    # Unbundle batch and index dimensions
    unbundled_shape = p.shape[:a] + indices.shape + p.shape[a + 1:]
    gathered = gathered.reshape(unbundled_shape)

    # Shift batch dimensions back to the left
    if a > 0:
        perm = list(range(len(unbundled_shape)))
        perm = perm[a:(a + b)] + perm[:a] + perm[(a + b):]
        gathered = gathered.transpose(perm)

    return gathered


def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
    if drop_mask_channel:
        mask = mask[:, 0]

    mask_shape = mask.shape
    value_shape = value.shape
    assert len(mask_shape) == len(value_shape)

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))

    assert isinstance(axis, collections.abc.Iterable), \
        'axis needs to be either an iterable, integer or "None"'

    broadcast_factor = 1.
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            assert mask_size == value_size

    return (paddle.sum(mask * value, axis=axis) /
            (paddle.sum(mask, axis=axis) * broadcast_factor + eps))


class Transition(nn.Layer):
    """Transition layer.

    Jumper et al. (2021) Suppl. Alg. 9 "MSATransition"
    Jumper et al. (2021) Suppl. Alg. 15 "PairTransition"
    """

    def __init__(self, channel_num, config, global_config, is_extra_msa,
                 transition_type):
        super(Transition, self).__init__()
        assert transition_type in ['msa_transition', 'pair_transition']
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.is_extra_msa = is_extra_msa
        self.transition_type = transition_type

        if transition_type == 'msa_transition' and is_extra_msa:
            in_dim = channel_num['extra_msa_channel']
        elif transition_type == 'msa_transition' and not is_extra_msa:
            in_dim = channel_num['msa_channel']
        elif transition_type == 'pair_transition':
            in_dim = channel_num['pair_channel']

        self.input_layer_norm = nn.LayerNorm(in_dim)
        self.transition1 = nn.Linear(
            in_dim,
            int(in_dim * self.config.num_intermediate_factor),
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingNormal()))

        if self.global_config.zero_init:
            last_init = nn.initializer.Constant(0.0)
        else:
            last_init = nn.initializer.TruncatedNormal()

        self.transition2 = nn.Linear(
            int(in_dim * self.config.num_intermediate_factor),
            in_dim,
            weight_attr=paddle.ParamAttr(initializer=last_init))

    def forward(self, act, mask):
        act = self.input_layer_norm(act)

        def transition_module(x):
            x = self.transition1(x)
            x = nn.functional.relu(x)
            x = self.transition2(x)
            return x

        if not self.training:
            # low memory mode using subbatch
            sb_transition = subbatch(transition_module, [0], [1],
                                     self.global_config.subbatch_size, 1)
            act = sb_transition(act)
        else:
            act = transition_module(act)

        return act


class Dropout(nn.Layer):
    def __init__(self, p=0.5, axis=None, mode="upscale_in_train", name=None):
        super(Dropout, self).__init__()

        if not isinstance(p, (float, int)):
            raise TypeError("p argument should be a number")
        if p < 0 or p > 1:
            raise ValueError("p argument should between 0 and 1")

        mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer
        if mode not in ('downscale_in_infer', 'upscale_in_train'):
            raise ValueError(
                "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
            )

        if axis and not isinstance(axis, (int, list, tuple)):
            raise TypeError("datatype of axis argument should be int or list")

        self.p = p
        self.axis = axis
        self.mode = mode
        self.name = name

    def forward(self, input):
        # fast return for p == 0
        if self.p == 0:
            return input

        if self.axis == None:
            out = nn.functional.dropout(
                input,
                p=self.p,
                axis=self.axis,
                training=self.training,
                mode=self.mode,
                name=self.name)
        else:
            seed = None
            drop_axes = [self.axis] if isinstance(self.axis,
                                                  int) else list(self.axis)
            if paddle.static.default_main_program().random_seed != 0:
                seed = paddle.static.default_main_program().random_seed

            out, mask = _C_ops.dropout_nd(
                input, 'dropout_prob', self.p, 'is_test', not self.training,
                'fix_seed', seed is not None, 'seed', seed if seed is not None
                else 0, 'dropout_implementation', self.mode, 'axis', drop_axes)

        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'p={}, axis={}, mode={}{}'.format(self.p, self.axis, self.mode,
                                                 name_str)


def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    lower_breaks = paddle.linspace(min_bin, max_bin, num_bins)
    lower_breaks = paddle.square(lower_breaks)
    upper_breaks = paddle.concat(
        [lower_breaks[1:], paddle.to_tensor(
            [1e8], dtype='float32')])

    def _squared_difference(x, y):
        return paddle.square(x - y)

    dist2 = paddle.sum(_squared_difference(
        paddle.unsqueeze(
            positions, axis=-2),
        paddle.unsqueeze(
            positions, axis=-3)),
                       axis=-1,
                       keepdim=True)

    dgram = ((dist2 > lower_breaks.astype(dist2.dtype)).astype('float32') *
             (dist2 < upper_breaks.astype(dist2.dtype)).astype('float32'))
    return dgram
