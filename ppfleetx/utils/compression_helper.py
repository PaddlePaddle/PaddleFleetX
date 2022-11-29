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

import paddle
import paddleslim


def get_pruned_params(model):
    params = []
    for sublayer in model.sublayers():
        for param in sublayer.parameters(include_sublayers=False):
            if isinstance(sublayer,
                          paddle.nn.layer.common.Linear) or isinstance(
                              sublayer, paddle.distributed.fleet.layers.mpu.
                              mp_layers.ColumnParallelLinear) or isinstance(
                                  sublayer, paddle.distributed.fleet.layers.
                                  mpu.mp_layers.RowParallelLinear):
                if len(param.shape) != 2: continue
                if param.shape[1] == 3 * param.shape[0] or param.shape[
                        1] == 4 * param.shape[0]:
                    params.append(param.name)

    return params


def prune_model(model, configs, num_attention_heads, infer=False):
    prune_criterion = configs.criterion
    ratio = configs.ratio

    if prune_criterion == 'l1_norm':
        if infer:
            pruner = paddleslim.dygraph.L1NormFilterPruner(
                model, [[1, 1024]],
                skip_leaves=False,
                prune_type='fc',
                input_dtype='int8',
                num_head=num_attention_heads)
        else:
            pruner = paddleslim.dygraph.L1NormFilterPruner(
                model, [[1, 1024], [1, 1024]],
                skip_leaves=False,
                prune_type='fc',
                input_dtype='int8',
                num_head=num_attention_heads)
    elif prune_criterion == 'l2_norm':
        if infer:
            pruner = paddleslim.dygraph.L2NormFilterPruner(
                model, [[1, 1024]],
                skip_leaves=False,
                prune_type='fc',
                input_dtype='int8',
                num_head=num_attention_heads)
        else:
            pruner = paddleslim.dygraph.L2NormFilterPruner(
                model, [[1, 1024], [1, 1024]],
                skip_leaves=False,
                prune_type='fc',
                input_dtype='int8',
                num_head=num_attention_heads)
    params = get_pruned_params(model)
    ratios = {}
    for param in params:
        ratios[param] = ratio
    #NOTE(minghaoBD): hidden size in Layernorm must be 768/1024/2048/4096 for best inference performace, and when axis=0, the hidden size in layernorm will be changed accordingly. So axis=1 is required.
    plan = pruner.prune_vars(ratios, [1])


def quant_model(model, configs):
    quanter = paddleslim.dygraph.quant.QAT(configs)
    return quanter.quantize(model), quanter
