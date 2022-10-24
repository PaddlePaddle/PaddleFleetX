# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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
from paddle.fluid.clip import ClipGradByGlobalNorm

from paddle.fluid.clip import ClipGradBase, _squared_l2_norm
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid import core, layers
from paddle.distributed import collective
import paddle.distributed.fleet as fleet


class ClipGradForMOEByGlobalNorm(ClipGradBase):
    def __init__(self, clip_norm):
        super(ClipGradForMOEByGlobalNorm, self).__init__()
        self.clip_norm = float(clip_norm)

        self.moe_group = None
        self.world_size = paddle.distributed.get_world_size()
        if self.world_size > 1:
            hcg = fleet.get_hybrid_communicate_group()
            self.moe_group = hcg.get_expert_parallel_group()

    def __str__(self):
        return "Gradient Clip By GlobalNorm, global_norm=%f" % (self.clip_norm)

    @staticmethod
    def get_l2_norm_pow(params_grads, sum_dtype=None):
        sum_square_list = []
        sum_square_list_fp16 = []
        sum_square_list_fp32 = []
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = layers.merge_selected_rows(g)
                merge_grad = layers.get_tensor_from_selected_rows(merge_grad)
            sum_square = _squared_l2_norm(merge_grad)
            if sum_square.dtype == core.VarDesc.VarType.FP16:
                sum_square_list_fp16.append(sum_square)
            elif sum_square.dtype == core.VarDesc.VarType.FP32:
                sum_square_list_fp32.append(sum_square)
            else:
                sum_square_list.append(sum_square)

        # all parameters have been filterd out
        if len(sum_square_list) + len(sum_square_list_fp16) + len(
                sum_square_list_fp32) == 0:
            return None, None
        assert sum_dtype in ["float64", "float32", None], \
            "sum's type must be float64/ float32 / None"
        if sum_dtype != "float64":
            sum_dtype = 'float64' if len(sum_square_list) > 0 else "float32"

        global_norm_var = []
        if len(sum_square_list_fp16) > 0:
            global_norm_var_fp16 = layers.concat(sum_square_list_fp16)
            global_norm_var_fp16 = layers.reduce_sum(global_norm_var_fp16)
            global_norm_var.append(global_norm_var_fp16.astype(sum_dtype))
        if len(sum_square_list_fp32) > 0:
            global_norm_var_fp32 = layers.concat(sum_square_list_fp32)
            global_norm_var_fp32 = layers.reduce_sum(global_norm_var_fp32)
            if sum_dtype == 'float32':
                global_norm_var.append(global_norm_var_fp32)
            else:
                global_norm_var.append(global_norm_var_fp32.astype(sum_dtype))
        if len(sum_square_list) > 0:
            global_norm_var_fp64 = layers.concat(sum_square_list)
            global_norm_var_fp64 = layers.reduce_sum(global_norm_var_fp64)
            global_norm_var.append(global_norm_var_fp64)
        global_norm_var = layers.concat(global_norm_var)
        global_norm_var = layers.reduce_sum(global_norm_var)
        return global_norm_var, sum_dtype

    @imperative_base.no_grad
    def _dygraph_clip(self, params_grads):
        normal_params_grads = []
        moe_params_grads = []

        # separate moe params from normal params
        if self.moe_group is not None and self.moe_group.nranks > 1:
            for p, g in params_grads:
                if "expert" in p.name or "gate" in p.name:
                    moe_params_grads.append((p, g))
                else:
                    normal_params_grads.append((p, g))
        else:
            normal_params_grads = params_grads

        # why to return sum_dtype?
        # we will call `get_l2_norm_pow` twice and the precisions may be different.
        # For convenience and simplification, we use sum_dtype directly instead of global_norm_var_normal.dtype
        global_norm_var_normal, sum_dtype \
            = self.get_l2_norm_pow(normal_params_grads)
        global_norm_var_moe = None
        if len(moe_params_grads) > 0:
            global_norm_var_moe, _ \
                = self.get_l2_norm_pow(moe_params_grads, sum_dtype)
            if global_norm_var_moe is not None:
                collective.all_reduce(
                    global_norm_var_moe,
                    op=collective.ReduceOp.SUM,
                    group=self.moe_group)

        if global_norm_var_normal is None and global_norm_var_moe is None:
            return params_grads
        elif global_norm_var_normal is None:
            global_norm_var = global_norm_var_moe
        elif global_norm_var_moe is None:
            global_norm_var = global_norm_var_normal
        else:
            if global_norm_var_normal.dtype != global_norm_var_moe.dtype:
                # compared with normal norm, moe norm is the later one,
                # so its precision is no lower than normal norm
                global_norm_var_normal = \
                    global_norm_var_normal.astype(global_norm_var_moe.dtype)
            global_norm_var = global_norm_var_normal + global_norm_var_moe

        global_norm_var = layers.sqrt(global_norm_var)
        max_global_norm = layers.fill_constant(
            shape=[1], dtype=global_norm_var.dtype, value=self.clip_norm)
        clip_var = layers.elementwise_div(
            x=max_global_norm,
            y=layers.elementwise_max(
                x=global_norm_var, y=max_global_norm))
        clip_var_fp16 = paddle.cast(clip_var, paddle.float16)

        for p, g in params_grads:
            if g is None or getattr(p, 'need_clip', True) is False:
                continue

            if p.dtype == paddle.float16:
                g.scale_(clip_var_fp16)
            else:
                g.scale_(clip_var)

            p._reset_grad_inplace_version(True)

        return params_grads
