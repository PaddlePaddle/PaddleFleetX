#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Transformer block."""

from functools import partial

import paddle
import paddle.fluid.layers as layers
import paddle.distributed.fleet as fleet
import paddle.nn as nn

from knover.modules.transformer_block import multi_head_attention, positionwise_feed_forward
from knover.modules.transformer_block import pre_process_layer, post_process_layer
from knover.modules.transformer_block import encoder_layer



def moe_positionwise_feed_forward(x,
                                  d_inner_hid,
                                  d_hid,
                                  num_experts,
                                  experts_capacity,
                                  dropout_rate,
                                  hidden_act,
                                  padding_mask=None,
                                  param_initializer=None,
                                  name="ffn"):
    """MoE Position-wise Feed-Forward Networks.

    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    expert_logits = paddle.static.nn.fc(x=x,
                              size=num_experts,
                              num_flatten_dims=2,
                              weight_attr=paddle.ParamAttr(
                                  name=name + "_expert_gating.w_0",
                                  initializer=param_initializer),
                              bias_attr=False)
    selected_expert = paddle.argmax(expert_logits, axis=2)
    selected_expert = paddle.unsqueeze(selected_expert, axis=[2])

    # Calculate expert mask.
    expert_mask = layers.one_hot(selected_expert, num_experts)
    if padding_mask is not None:
        expert_mask = expert_mask * (1 - padding_mask)

    # calculate auxiliary loss
    num_tokens = paddle.sum(1 - padding_mask)
    density = paddle.sum(paddle.sum(expert_mask, axis=1), axis=0)
    density = density / num_tokens
    expert_probs = paddle.nn.functional.softmax(expert_logits)
    mask = paddle.sum(expert_mask, axis=2, keepdim=True)
    density_proxy = paddle.sum(paddle.sum(expert_probs * mask, axis=1), axis=0)
    density_proxy = density_proxy / num_tokens
    aux_loss = paddle.mean(density * density_proxy) * num_experts * num_experts

    # Handle over capacity.
    position_in_expert_0 = layers.cumsum(paddle.sum(expert_mask, axis=1, keepdim=True),
                                         axis=0, exclusive=True)
    position_in_expert_1 = layers.cumsum(expert_mask, axis=1, exclusive=True)
    position_in_expert = position_in_expert_0 + position_in_expert_1
    # filter tokens which exceed the selcted expert's capacity
    expert_mask *= position_in_expert < experts_capacity

    # flag whether the token is dispatched to one expert.
    mask = paddle.sum(expert_mask, axis=2, keepdim=True)
    # [B, S, 1]: flag whether the token is passed to next layer directly.
    stay_mask = 1 - mask

    # token embeddings in stay (position && number of tokens)
    position_in_stay_0 = layers.cumsum(paddle.sum(stay_mask, axis=1, keepdim=True),
                                       axis=0, exclusive=True)
    position_in_stay_1 = layers.cumsum(stay_mask, axis=1, exclusive=True)
    position_in_stay = position_in_stay_0 + position_in_stay_1
    # add padding in case no token is stay
    num_tokens_is_stay = position_in_stay[-1, -1] + 1

    # token embeddings in experts (position && number of tokens)
    num_tokens_in_each_experts = paddle.sum(
        paddle.sum(expert_mask, axis=0, keepdim=True),
        axis=1, keepdim=True
    )

    output_offset_for_expert = layers.cumsum(
        paddle.full(shape=[1, 1, num_experts], dtype="float32", fill_value=experts_capacity), axis=2, exclusive=True) + num_tokens_is_stay

    position_in_output_for_expert = position_in_expert + output_offset_for_expert
    position_in_output_for_expert = position_in_output_for_expert * expert_mask
    position_in_output_for_expert = paddle.sum(position_in_output_for_expert, axis=2, keepdim=True)

    position_in_output = mask * position_in_output_for_expert + stay_mask * position_in_stay
    position_in_output = paddle.cast(position_in_output, dtype="int64")

    selected_expert = selected_expert * mask - (1 - mask)
    selected_expert = paddle.squeeze(selected_expert, axis=[2])

    # dispatch input embeddings
    expert_inputs = []
    for expert_id in range(num_experts):
        idx = paddle.nonzero(selected_expert == expert_id)
        # padding
        idx = paddle.concat([idx, paddle.zeros([experts_capacity, 2], "int64")], axis=0)
        idx = idx[:experts_capacity]
        expert_input = paddle.gather_nd(x, idx)
        expert_inputs.append(expert_input)

    # get current expert's input
    the_expert_input = []
    paddle.distributed.alltoall(expert_inputs, the_expert_input)
    the_expert_input = paddle.concat(the_expert_input, axis=0)

    main_block = paddle.static.default_main_program().global_block()
    startup_block = paddle.static.default_startup_program().global_block()

    # run feed-forward network
    expert_id = fleet.worker_index()
    fc0 = nn.Linear(
        d_hid,
        d_inner_hid,
        weight_attr=paddle.ParamAttr(name=f"{name}_fc_0_e{expert_id}.w_0", initializer=param_initializer),
        bias_attr=paddle.ParamAttr(name=f"{name}_fc_0_e{expert_id}.b_0")
    )
    fc0.weight.is_distributed = True
    fc0.bias.is_distributed = True
    hidden = fc0(the_expert_input)

    if dropout_rate:
        hidden = paddle.nn.functional.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)

    fc1 = nn.Linear(
        d_inner_hid,
        d_hid,
        weight_attr=paddle.ParamAttr(name=f"{name}_fc_1_e{expert_id}.w_0", initializer=param_initializer),
        bias_attr=paddle.ParamAttr(name=f"{name}_fc_1_e{expert_id}.b_0")
    )
    fc1.weight.is_distributed = True
    fc1.bias.is_distributed = True
    the_expert_output = fc1(hidden)

    # dispatch output embeddings
    expert_outputs = []
    # expert tokens
    paddle.distributed.alltoall(paddle.split(the_expert_output, num_experts), expert_outputs)

    # stay tokens
    idx = paddle.nonzero(selected_expert == -1)
    idx = paddle.concat([idx, paddle.zeros([1, 2], "int64")], axis=0)
    out = paddle.gather_nd(x, idx)
    expert_outputs.insert(0, out)

    expert_outputs = paddle.concat(expert_outputs, axis=0)
    out = paddle.gather_nd(expert_outputs, position_in_output)
    return out, aux_loss


def moe_encoder_layer(input,
                      attn_bias,
                      n_head,
                      d_key,
                      d_value,
                      d_model,
                      d_inner_hid,
                      num_experts,
                      experts_capacity,
                      prepostprocess_dropout,
                      attention_dropout,
                      relu_dropout,
                      hidden_act,
                      padding_mask=None,
                      preprocess_cmd="n",
                      postprocess_cmd="da",
                      param_initializer=None,
                      name="",
                      epsilon=1e-5,
                      cache=None,
                      gather_idx=None,
                      store=False):
    """A MoE Transformer encoder block.

    The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components are companied
    with the pre_process_layer / post_process_layer to add residual connection,
    layer normalization and droput.
    """
    attn_output = multi_head_attention(
        pre_process_layer(
            input,
            preprocess_cmd,
            prepostprocess_dropout,
            epsilon=epsilon,
            name=name + "_pre_att"),
        None,
        None,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + "_multi_head_att",
        cache=cache,
        gather_idx=gather_idx,
        store=store)
    attn_output = post_process_layer(
        input,
        attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + "_post_att",
        epsilon=epsilon)
    ffd_output, aux_loss = moe_positionwise_feed_forward(
        pre_process_layer(
            attn_output,
            preprocess_cmd,
            prepostprocess_dropout,
            epsilon=epsilon,
            name=name + "_pre_ffn"),
        d_inner_hid,
        d_model,
        num_experts,
        experts_capacity,
        relu_dropout,
        hidden_act,
        padding_mask=padding_mask,
        param_initializer=param_initializer,
        name=name + "_ffn")
    ffd_output = post_process_layer(
        attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + "_post_ffn",
        epsilon=epsilon)
    return ffd_output, [ffd_output], aux_loss


def moe_encoder(enc_input,
                attn_bias,
                n_layer,
                n_head,
                d_key,
                d_value,
                d_model,
                d_inner_hid,
                num_experts,
                experts_capacity,
                prepostprocess_dropout,
                attention_dropout,
                relu_dropout,
                hidden_act,
                padding_mask=None,
                pre_encoder_cmd="nd",
                preprocess_cmd="n",
                postprocess_cmd="da",
                param_initializer=None,
                name="encoder",
                epsilon=1e-5,
                n_layer_per_block=1,
                param_share="normal",
                caches=None,
                gather_idx=None,
                store=False):
    """A MoE Transformer Encoder.

    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    sum_of_aux_loss = None
    checkpoints = []
    names = []
    if param_share == "inner_share":
        for _ in range(n_layer // n_layer_per_block):
            for i in range(n_layer_per_block):
                names.append(name + "_layer_" + str(i))
    else:
        for i in range(n_layer // n_layer_per_block):
            for _ in range(n_layer_per_block):
                names.append(name + "_layer_" + str(i))

    enc_input = pre_process_layer(
        enc_input,
        pre_encoder_cmd,
        prepostprocess_dropout,
        name=f"pre_{name}",
        epsilon=epsilon)
    for i in range(n_layer):
        if i % 2 == 1:
            enc_output, cps = encoder_layer(
                enc_input,
                attn_bias,
                n_head,
                d_key,
                d_value,
                d_model,
                d_inner_hid,
                prepostprocess_dropout,
                attention_dropout,
                relu_dropout,
                hidden_act,
                preprocess_cmd,
                postprocess_cmd,
                param_initializer=param_initializer,
                epsilon=epsilon,
                name=names[i],
                cache=caches[i] if caches is not None else None,
                gather_idx=gather_idx,
                store=store)
        else:
            enc_output, cps, aux_loss = moe_encoder_layer(
                enc_input,
                attn_bias,
                n_head,
                d_key,
                d_value,
                d_model,
                d_inner_hid,
                num_experts,
                experts_capacity,
                prepostprocess_dropout,
                attention_dropout,
                relu_dropout,
                hidden_act,
                padding_mask,
                preprocess_cmd,
                postprocess_cmd,
                param_initializer=param_initializer,
                epsilon=epsilon,
                name=names[i],
                cache=caches[i] if caches is not None else None,
                gather_idx=gather_idx,
                store=store)
            if aux_loss is not None:
                if sum_of_aux_loss is None:
                    sum_of_aux_loss = aux_loss
                else:
                    sum_of_aux_loss = sum_of_aux_loss + aux_loss
        checkpoints.extend(cps)
        enc_input = enc_output
    enc_output = pre_process_layer(
        enc_output,
        preprocess_cmd,
        prepostprocess_dropout,
        name=f"post_{name}",
        epsilon=epsilon)

    return enc_output, checkpoints, sum_of_aux_loss
