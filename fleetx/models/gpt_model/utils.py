# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
from paddle import framework
from paddle.nn import functional as F
from paddle.autograd import PyLayer
from paddle.fluid import core
from paddle.fluid.dygraph.layers import Layer
from paddle.distributed import fleet
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

import numpy as np



####################################################
#                                                  #
#        Distributed Communication Operator        #
#                                                  #
####################################################

class ScatterOp(PyLayer):
    # input shape: [s, b, h], n is mp parallelism
    # after forward shape: [s/n, b, h]
    @staticmethod
    def forward(self, input):
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
        parallelism = group.nranks
        rank = group.rank
        assert input.shape[0] % parallelism == 0, \
            "Input sequence length {0} can't be divided exactly \
             by sequence parallelism {1}".format(input.shape[0], parallelism)
        input = paddle.split(input, num_or_sections=parallelism, axis=0)[rank]
        return input

    @staticmethod
    def backward(self, grad):
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
        parallelism = group.nranks
        output_shape = grad.shape
        output_shape[0] = output_shape[0] * parallelism
        output = paddle.empty(shape=output_shape, dtype=grad.dtype)
        group.process_group.all_gather(grad, output).wait()
        return output

class GatherOp(PyLayer):
    # input shape: [s/n, b, h], n is mp parallelism
    # after forward shape: [s, b, h]
    @staticmethod
    def forward(self, input):
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
        parallelism = group.nranks
        output_shape = input.shape
        output_shape[0] = output_shape[0] * parallelism
        output = paddle.empty(shape=output_shape, dtype=input.dtype)
        group.process_group.all_gather(input, output).wait()
        return output

    @staticmethod
    def backward(self, grad):
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
        parallelism = group.nranks
        rank = group.rank
        assert grad.shape[0] % parallelism == 0, \
            "Input sequence length {0} can't be divided exactly \
             by sequence parallelism {1}".format(grad.shape[0], parallelism)
        input = paddle.split(grad, num_or_sections=parallelism, axis=0)[rank]
        return input

# All gather along the first dim during forward pass
# All reduce and scatter along the first dim during backward pass
class AllGatherOp(PyLayer):
    # input shape: [s/n, b, h], n is mp parallelism
    # after forward shape: [s, b, h]
    @staticmethod
    def forward(self, input):
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
        parallelism = group.nranks
        output_shape = input.shape
        output_shape[0] = output_shape[0] * parallelism
        output = paddle.empty(shape=output_shape, dtype=input.dtype)
        group.process_group.all_gather(input, output).wait()
        return output

    # grad shape: [s, b, h], n is mp parallelism
    # after forward shape: [s/n, b, h]
    @staticmethod
    def backward(self, grad):
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
        parallelism = group.nranks
        output_shape = grad.shape
        assert grad.shape[0] % parallelism == 0, \
            "Grad sequence length {0} can't be divided exactly by \
             sequence parallelism {1}".format(grad.shape[0], parallelism)
        output_shape[0] = output_shape[0] // parallelism
        output = paddle.empty(shape=output_shape, dtype=grad.dtype)
        group.process_group._reduce_scatter_base(output, grad).wait()
        return output

# All reduce and scatter along the first dim during forward pass
# All gather along the first dim during backward pass
class ReduceScatterOp(PyLayer):
    # input shape: [s, b, h], n is mp parallelism
    # after forward shape: [s/n, b, h]
    @staticmethod
    def forward(self, input):
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
        parallelism = group.nranks
        output_shape = input.shape
        assert input.shape[0] % parallelism == 0, \
            "Input sequence length {0} can't be divided exactly by \
             sequence parallelism {1}".format(input.shape[0], parallelism)
        output_shape[0] = output_shape[0] // parallelism
        output = paddle.empty(shape=output_shape, dtype=input.dtype)
        group.process_group._reduce_scatter_base(output, input).wait()
        return output

    # grad shape: [s/n, b, h], n is mp parallelism
    # after forward shape: [s, b, h]
    @staticmethod
    def backward(self, grad):
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
        parallelism = group.nranks
        output_shape = grad.shape
        output_shape[0] = output_shape[0] * parallelism
        output = paddle.empty(shape=output_shape, dtype=grad.dtype)
        group.process_group.all_gather(grad, output).wait()
        return output



###################################################
#                                                 #
#        Modified Parallel Linear Operator        #
#                                                 #
###################################################

def all_reduce_gradient_hook(grad):
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()
    group.process_group.allreduce(grad).wait()
    return grad

# def gradient_hook(grad, name="gradient", prefix="seq_paral", is_break=True):
#     hcg = fleet.get_hybrid_communicate_group()
#     group = hcg.get_model_parallel_group()
#     rank = group.rank
#     np.save("debug_data/{0}/{1}_{2}.npy".format(prefix, name, rank), grad)
#     if is_break:
#         import os; os._exit(0)
#     return grad

def is_fused_matmul_bias_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
        return hasattr(core.ops, 'fused_gemm_epilogue')
    else:
        return False

class ColumnSequenceParallelLinear(Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 has_bias=None,
                 gather_output=True,
                 fuse_matmul_bias=False,
                 mp_group=None,
                 name=None):
        super(ColumnSequenceParallelLinear, self).__init__()

        self.model_parallel_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group(
        ) if mp_group is None else mp_group
        self.world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size(
        ) if mp_group is None else mp_group.nranks
        self._name = name
        self.is_mp = (self.world_size > 1)

        assert gather_output is False, "If sequence_parallel is True, \
                                        gather_output is False"
        self.gather_output = gather_output
        assert out_features % self.world_size == 0, (
            "Number of column of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(
                out_features, self.world_size))
        self.output_size_per_partition = out_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[in_features, self.output_size_per_partition],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False)
        else:
            self.weight = self.create_parameter(
                shape=[in_features, self.output_size_per_partition],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False)

        self.weight.is_distributed = True if self.is_mp else False

        if has_bias:
            # initialize bias to zero like Megatron
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True)
            self.bias.is_distributed = True if self.is_mp else False
        else:
            self.bias = None

        self.linear = F.linear

        if fuse_matmul_bias:
            if not is_fused_matmul_bias_supported():
                raise NotImplementedError(
                    "You set fuse_matmul_bias=True in ColumnSequenceParallelLinear, "
                    "however, the paddle you are using not support this operation. "
                    "Please set fuse_matmul_bias=False or use paddle compiled "
                    "with cuda 11.6 or higher.")
            from paddle.incubate.nn.functional import fused_linear
            self.linear = fused_linear


    def forward(self, x):
        # sequence parallelism is same as model parallelism
        # if sequence parallel is true, input shape is [s, b, h]
        # else input shape is [b, s, h]
        if self.is_mp:
            input_parallel = AllGatherOp.apply(x)
        else:
            input_parallel = x
        output = self.linear(input_parallel,
                                      self.weight,
                                      self.bias,
                                      name=self._name)
        return output


class RowSequenceParallelLinear(Layer):

    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 has_bias=True,
                 input_is_parallel=False,
                 fuse_matmul_bias=False,
                 mp_group=None,
                 name=None):
        super(RowSequenceParallelLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert input_is_parallel is True, "If sequence_parallel is True, \
                                           input_is_parallel should be true."
        self.input_is_parallel = input_is_parallel
        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()
        self._name = name

        self.model_parallel_group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group(
        ) if mp_group is None else mp_group
        self.world_size = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size(
        ) if mp_group is None else mp_group.nranks
        self.rank = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank(
        ) if mp_group is None else mp_group.rank

        self.is_mp = (self.world_size > 1)
        assert in_features % self.world_size == 0, (
            "Number of row of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(
                in_features, self.world_size))

        self.input_size_per_partition = in_features // self.world_size

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[self.input_size_per_partition, self.out_features],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False)
        else:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, self.out_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False)

        self.weight.is_distributed = True if self.is_mp else False

        # if sequence parallel is true, 
        # register hook to all_reduce gradient of weight and bias
        # if self.is_mp:
        #     self.weight.register_hook(all_reduce_gradient_hook)
        if has_bias:
            self.bias = self.create_parameter(
                shape=[self.out_features],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True)
            if self.is_mp:
                self.bias.register_hook(all_reduce_gradient_hook)
        else:
            self.bias = None

        self.linear = F.linear

        if fuse_matmul_bias:
            if not is_fused_matmul_bias_supported():
                raise NotImplementedError(
                    "You set fuse_matmul_bias=True in RowParallelLinear, "
                    "however, the paddle you are using not support this operation. "
                    "Please set fuse_matmul_bias=False or use paddle compiled "
                    "with cuda 11.6 or higher.")
            from paddle.incubate.nn.functional import fused_linear
            self.linear = fused_linear

    def forward(self, x):
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = x
        else:
            # split last dim
            input_parallel = paddle.distributed.collective._c_split(
                x, group=self.model_parallel_group)

        if self.is_mp:
            output_parallel = self.linear(input_parallel,
                                          self.weight,
                                          name=self._name)
            output_ = ReduceScatterOp.apply(output_parallel)
            # if self.bias is not none, sequence parallel will use
            # register_hook to all_reduce self.bias
            output = output_ + self.bias if self.bias is not None else output_
        else:
            output = self.linear(input_parallel,
                                 self.weight,
                                 self.bias,
                                 name=self._name)
        return output
