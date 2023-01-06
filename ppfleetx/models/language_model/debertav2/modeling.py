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
""" Paddle DeBERTa-v2 model."""

from collections.abc import Sequence
from typing import Optional, Tuple, Union
import json

import paddle
from paddle import nn
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

from ppfleetx.models.language_model.t5 import (finfo, ACT2FN, ModelOutput,
                                               normal_, constant_init)
from ppfleetx.data.tokenizers.debertav2_tokenizer import debertav2_tokenize

from dataclasses import dataclass


class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state = None
    hidden_states = None
    attentions = None


# Copied from transformers.models.deberta.modeling_deberta.XSoftmax with deberta->deberta_v2
class XSoftmax(paddle.autograd.PyLayer):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`paddle.tensor`): The input tensor that will apply softmax.
        mask (`paddle.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example:

    ```python
    >>> import paddle 
    >>> from transformers.models.deberta_v2.modeling_deberta_v2 import XSoftmax

    >>> # Make a tensor
    >>> x = paddle.randn([4, 20, 100])

    >>> # Create a mask
    >>> mask = (x > 0).int()

    >>> # Specify the dimension to apply softmax
    >>> dim = -1

    >>> y = XSoftmax.apply(x, mask, dim)
    ```"""

    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        #rmask = ~(mask.cast('bool'))
        #output = input.masked_fill(rmask, paddle.to_tensor(finfo(input.dtype).min))
        mask = mask.cast('bool')
        output = paddle.where(mask == 0,
                              paddle.to_tensor(finfo(input.dtype).min), input)
        output = paddle.nn.functional.softmax(
            output, axis=self.dim, dtype=paddle.float32)
        output = paddle.where(mask == 0, paddle.to_tensor(0.), output)
        return output


# Copied from transformers.models.deberta.modeling_deberta.DropoutContext
class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


# Copied from transformers.models.deberta.modeling_deberta.get_mask
def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - paddle.bernoulli(
            paddle.full(
                shape=input.shape, fill_value=1 - dropout))).cast(bool)

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


# Copied from transformers.models.deberta.modeling_deberta.XDropout
class XDropout(paddle.autograd.PyLayer):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            output = paddle.where(mask == 1, 0, input)
            return output * ctx.scale
        else:
            return input


# Copied from transformers.models.deberta.modeling_deberta.StableDropout
class StableDropout(nn.Layer):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (`paddle.to_tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


# Copied from transformers.models.deberta.modeling_deberta.DebertaSelfOutput with DebertaLayerNorm->LayerNorm
class DebertaV2SelfOutput(nn.Layer):
    def __init__(self,
                 hidden_size=1536,
                 layer_norm_eps=1e-7,
                 hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = StableDropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaAttention with Deberta->DebertaV2
class DebertaV2Attention(nn.Layer):
    def __init__(
            self,
            hidden_size=512,
            num_attention_heads=24,
            attention_head_size=64,
            share_att_key=True,
            pos_att_type=None,
            relative_attention=True,
            position_buckets=-1,
            max_relative_positions=-1,
            max_position_embeddings=512,
            layer_norm_eps=1e-7,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1, ):
        super().__init__()
        self.self = DisentangledSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_size=attention_head_size,
            share_att_key=share_att_key,
            pos_att_type=pos_att_type,
            relative_attention=relative_attention,
            position_buckets=position_buckets,
            max_relative_positions=max_relative_positions,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob, )
        self.output = DebertaV2SelfOutput(
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob)

    def forward(
            self,
            hidden_states,
            attention_mask,
            output_attentions=False,
            query_states=None,
            relative_pos=None,
            rel_embeddings=None, ):
        self_output = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings, )
        if output_attentions:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return attention_output


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->DebertaV2
class DebertaV2Intermediate(nn.Layer):
    def __init__(
            self,
            hidden_size=1536,
            hidden_act='gelu',
            intermediate_size=6144, ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaOutput with DebertaLayerNorm->LayerNorm
class DebertaV2Output(nn.Layer):
    def __init__(
            self,
            hidden_size=512,
            intermediate_size=6144,
            layer_norm_eps=1e-7,
            hidden_dropout_prob=0.1, ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = StableDropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaLayer with Deberta->DebertaV2
class DebertaV2Layer(nn.Layer):
    def __init__(
            self,
            hidden_size=512,
            hidden_act='gelu',
            intermediate_size=6144,
            num_attention_heads=24,
            attention_head_size=64,
            share_att_key=True,
            pos_att_type=None,
            relative_attention=True,
            position_buckets=256,
            max_relative_positions=-1,
            max_position_embeddings=512,
            layer_norm_eps=1e-7,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1, ):
        super().__init__()
        self.attention = DebertaV2Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_size=attention_head_size,
            share_att_key=share_att_key,
            pos_att_type=pos_att_type,
            relative_attention=relative_attention,
            position_buckets=position_buckets,
            max_relative_positions=max_relative_positions,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob, )
        self.intermediate = DebertaV2Intermediate(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            intermediate_size=intermediate_size, )
        self.output = DebertaV2Output(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob, )

    def forward(
            self,
            hidden_states,
            attention_mask,
            query_states=None,
            relative_pos=None,
            rel_embeddings=None,
            output_attentions=False, ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings, )
        if output_attentions:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output


class ConvLayer(nn.Layer):
    def __init__(
            self,
            hidden_size=512,
            conv_kernel_size=3,
            conv_groups=1,
            conv_act="tanh",
            layer_norm_eps=1e-7,
            hidden_dropout_prob=0., ):
        super().__init__()
        kernel_size = conv_kernel_size
        groups = conv_groups
        self.conv_act = conv_act
        self.conv = nn.Conv1D(
            hidden_size,
            hidden_size,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=groups)
        self.LayerNorm = LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = StableDropout(hidden_dropout_prob)

    def forward(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.transpose([0, 2, 1])).transpose(
            [0, 2, 1])
        out = paddle.where(
            input_mask.cast('bool').unsqueeze(-1).expand(out.shape) == 0,
            paddle.to_tensor(0.), out)
        out = ACT2FN[self.conv_act](self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input).cast(layer_norm_input.dtype)

        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask.cast(output.dtype)
            output_states = output * input_mask

        return output_states


class DebertaV2Encoder(nn.Layer):
    """Modified BertEncoder with relative position bias support"""

    def __init__(
            self,
            num_hidden_layers=48,
            num_attention_heads=24,
            attention_head_size=64,
            relative_attention=False,
            max_relative_positions=-1,
            max_position_embeddings=512,
            position_buckets=256,
            hidden_size=1536,
            hidden_act='gelu',
            conv_act='gelu',
            intermediate_size=6144,
            share_att_key=True,
            pos_att_type=None,
            norm_rel_ebd=None,
            conv_kernel_size=0,
            layer_norm_eps=1e-7,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1, ):
        super().__init__()

        self.layer = nn.LayerList([
            DebertaV2Layer(
                hidden_size=hidden_size,
                hidden_act=hidden_act,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                attention_head_size=attention_head_size,
                share_att_key=share_att_key,
                pos_att_type=pos_att_type,
                relative_attention=relative_attention,
                position_buckets=position_buckets,
                max_relative_positions=max_relative_positions,
                max_position_embeddings=max_position_embeddings,
                layer_norm_eps=layer_norm_eps,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob)
            for _ in range(num_hidden_layers)
        ])
        self.relative_attention = relative_attention

        if self.relative_attention:
            self.max_relative_positions = max_relative_positions
            if self.max_relative_positions < 1:
                self.max_relative_positions = max_position_embeddings

            self.position_buckets = position_buckets
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = nn.Embedding(pos_ebd_size, hidden_size)

        self.norm_rel_ebd = [
            x.strip() for x in norm_rel_ebd.lower().split("|")
        ]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(hidden_size, layer_norm_eps)

        self.conv = ConvLayer(
            hidden_size=hidden_size,
            conv_kernel_size=conv_kernel_size,
            conv_act=conv_act,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
        ) if conv_kernel_size > 0 else None
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(
                -2).unsqueeze(-1)
            attention_mask = attention_mask.cast(paddle.uint8)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.shape[
                -2] if query_states is not None else hidden_states.shape[-2]
            relative_pos = build_relative_position(
                q,
                hidden_states.shape[-2],
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions)
        return relative_pos

    def forward(
            self,
            hidden_states,
            attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            query_states=None,
            relative_pos=None,
            return_dict=True, ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).cast(paddle.uint8)
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states,
                                        relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states, )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                output_states = paddle.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings, )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions, )

            if output_attentions:
                output_states, att_m = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states,
                                          input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(
                        self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m, )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states, )

        if not return_dict:
            return tuple(
                v for v in [output_states, all_hidden_states, all_attentions]
                if v is not None)
        return BaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions)


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = paddle.sign(relative_pos.cast('float32'))
    mid = bucket_size // 2
    abs_pos = paddle.where(
        (relative_pos < mid) & (relative_pos > -mid),
        paddle.to_tensor(mid - 1).astype(relative_pos.dtype),
        paddle.abs(relative_pos), )
    log_pos = (paddle.ceil(
        paddle.log(abs_pos / mid) /
        paddle.log(paddle.to_tensor((max_position - 1) / mid)) *
        (mid - 1)) + mid)
    bucket_pos = paddle.where(abs_pos <= mid,
                              relative_pos.cast(log_pos.dtype), log_pos * sign)
    return bucket_pos


def build_relative_position(query_size,
                            key_size,
                            bucket_size=-1,
                            max_position=-1):
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position

    Return:
        `paddle.LongTensor`: A tensor with shape [1, query_size, key_size]

    """
    q_ids = paddle.arange(0, query_size)
    k_ids = paddle.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size,
                                               max_position)
    rel_pos_ids = rel_pos_ids.cast(paddle.int64)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


# Copied from transformers.models.deberta.modeling_deberta.c2p_dynamic_expand
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([
        query_layer.shape[1], query_layer.shape[1], query_layer.shape[2],
        relative_pos.shape[-1]
    ])


# Copied from transformers.models.deberta.modeling_deberta.p2c_dynamic_expand
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([
        query_layer.shape[0], query_layer.shape[1], key_layer.shape[-2],
        key_layer.shape[-2]
    ])


# Copied from transformers.models.deberta.modeling_deberta.pos_dynamic_expand
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand([
        tuplt(p2c_att.shape[:2]) + (pos_index.shape[-2], key_layer.shape[-2])
    ])


class DisentangledSelfAttention(nn.Layer):
    """
    Disentangled self-attention module

    Parameters:

    """

    def __init__(
            self,
            hidden_size=1536,
            num_attention_heads=24,
            attention_head_size=None,
            share_att_key=False,
            pos_att_type=None,
            relative_attention=False,
            position_buckets=-1,
            max_relative_positions=-1,
            max_position_embeddings=512,
            hidden_dropout_prob=0.,
            attention_probs_dropout_prob=0., ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})")
        self.num_attention_heads = num_attention_heads
        _attention_head_size = hidden_size // num_attention_heads
        self.attention_head_size = attention_head_size if attention_head_size is not None else _attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(hidden_size, self.all_head_size)
        self.key_proj = nn.Linear(hidden_size, self.all_head_size)
        self.value_proj = nn.Linear(hidden_size, self.all_head_size)

        self.share_att_key = share_att_key
        self.pos_att_type = pos_att_type if pos_att_type is not None else []
        self.relative_attention = relative_attention

        if self.relative_attention:
            self.position_buckets = position_buckets
            self.max_relative_positions = max_relative_positions
            if self.max_relative_positions < 1:
                self.max_relative_positions = max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = StableDropout(hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(
                        hidden_size, self.all_head_size, bias=True)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(hidden_size,
                                                    self.all_head_size)

        self.dropout = StableDropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = tuple(x.shape[:-1]) + (attention_heads, -1)
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3]).reshape([-1, x.shape[1], x.shape[-1]])

    def forward(
            self,
            hidden_states,
            attention_mask,
            output_attentions=False,
            query_states=None,
            relative_pos=None,
            rel_embeddings=None, ):
        """
        Call the module

        Args:
            hidden_states (`paddle.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`paddle.uint8`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            output_attentions (`bool`, optional):
                Whether return the attention matrix.

            query_states (`paddle.FloatTensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`paddle.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`paddle.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(
            self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(
            self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(
            self.value_proj(hidden_states), self.num_attention_heads)

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = paddle.sqrt(
            paddle.to_tensor(
                query_layer.shape[-1], dtype='float32') * scale_factor)
        attention_scores = paddle.bmm(
            query_layer, key_layer.transpose(
                [0, 2, 1])) / scale.cast(query_layer.dtype)
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings,
                scale_factor)

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores
        attention_scores = attention_scores.reshape([
            -1, self.num_attention_heads, attention_scores.shape[-2],
            attention_scores.shape[-1]
        ])

        # bsz x height x length x dimension
        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        context_layer = paddle.bmm(
            attention_probs.reshape(
                [-1, attention_probs.shape[-2], attention_probs.shape[-1]]),
            value_layer)
        context_layer = (context_layer.reshape([
            -1, self.num_attention_heads, context_layer.shape[-2],
            context_layer.shape[-1]
        ]).transpose([0, 2, 1, 3]))
        new_context_layer_shape = tuple(context_layer.shape[:-2]) + (-1, )
        context_layer = context_layer.reshape(new_context_layer_shape)
        if output_attentions:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos,
                                    rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.shape[-2]
            relative_pos = build_relative_position(
                q,
                key_layer.shape[-2],
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bsz x height x query x key
        elif relative_pos.dim() != 4:
            raise ValueError(
                f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}"
            )

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.cast(paddle.int64)

        rel_embeddings = rel_embeddings[0:att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = paddle.tile(
                self.transpose_for_scores(
                    self.query_proj(rel_embeddings), self.num_attention_heads),
                repeat_times=[
                    query_layer.shape[0] // self.num_attention_heads, 1, 1
                ])
            pos_key_layer = paddle.tile(
                self.transpose_for_scores(
                    self.key_proj(rel_embeddings), self.num_attention_heads),
                repeat_times=[
                    query_layer.shape[0] // self.num_attention_heads, 1, 1
                ])
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = paddle.tile(
                    self.transpose_for_scores(
                        self.pos_key_proj(rel_embeddings),
                        self.num_attention_heads),
                    repeat_times=[
                        query_layer.shape[0] // self.num_attention_heads, 1, 1
                    ])  # .split(self.all_head_size, dim=-1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = paddle.tile(
                    self.transpose_for_scores(
                        self.pos_query_proj(rel_embeddings),
                        self.num_attention_heads),
                    repeat_times=[
                        query_layer.shape[0] // self.num_attention_heads, 1, 1
                    ])  # .split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = paddle.sqrt(
                paddle.to_tensor(
                    pos_key_layer.shape[-1], dtype='float32') * scale_factor)
            c2p_att = paddle.bmm(query_layer,
                                 pos_key_layer.transpose([0, 2, 1]))
            c2p_pos = paddle.clip(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = paddle.take_along_axis(
                c2p_att,
                axis=-1,
                indices=c2p_pos.squeeze(0).expand([
                    query_layer.shape[0], query_layer.shape[1],
                    relative_pos.shape[-1]
                ]), )
            score += c2p_att / scale.cast(dtype=c2p_att.dtype)

        # position->content
        if "p2c" in self.pos_att_type:
            scale = paddle.sqrt(
                paddle.to_tensor(
                    pos_query_layer.shape[-1], dtype='float32') * scale_factor)
            if key_layer.shape[-2] != query_layer.shape[-2]:
                r_pos = build_relative_position(
                    key_layer.shape[-2],
                    key_layer.shape[-2],
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions, )
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = paddle.clip(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = paddle.bmm(key_layer,
                                 pos_query_layer.transpose([0, 2, 1]))
            p2c_att = paddle.take_along_axis(
                p2c_att,
                axis=-1,
                indices=p2c_pos.squeeze(0).expand([
                    query_layer.shape[0], key_layer.shape[-2],
                    key_layer.shape[-2]
                ]), ).transpose([0, 2, 1])
            score += p2c_att / scale.cast(dtype=p2c_att.dtype)

        return score


# Copied from transformers.models.deberta.modeling_deberta.DebertaEmbeddings with DebertaLayerNorm->LayerNorm
class DebertaV2Embeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(
            self,
            max_position_embeddings=512,
            position_biased_input=False,
            pad_token_id=0,
            hidden_size=1536,
            hidden_dropout_prob=0.1,
            embedding_size=None,
            vocab_size=128100,
            type_vocab_size=0,
            layer_norm_eps=1e-7, ):
        super().__init__()
        self.embedding_size = hidden_size if embedding_size is None else embedding_size
        self.word_embeddings = nn.Embedding(
            vocab_size, self.embedding_size, padding_idx=pad_token_id)
        self.type_vocab_size = type_vocab_size
        self.hidden_size = hidden_size

        self.position_biased_input = position_biased_input
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                    self.embedding_size)

        if type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                      self.embedding_size)

        if self.embedding_size != hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = StableDropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids",
                             paddle.arange(max_position_embeddings).expand(
                                 (1, -1)))

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                mask=None,
                inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(
                position_ids.cast(paddle.int64))
        else:
            position_embeddings = paddle.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embedding_size != self.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.cast('float32')

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.deberta.modeling_deberta.DebertaPreTrainedModel with Deberta->DebertaV2
class DebertaV2PreTrainedModel(nn.Layer):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    base_model_prefix = "deberta"
    _keys_to_ignore_on_load_missing = ["position_ids"]
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                constant_init(module.bias, 0.)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                constant_init(module.weight.data[module.padding_idx], 0.)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, DebertaV2Encoder):
            module.gradient_checkpointing = value


DEBERTA_START_DOCSTRING = r"""
    The DeBERTa model was proposed in [DeBERTa: Decoding-enhanced BERT with Disentangled
    Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build
    on top of BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

    This model is also a PyTorch [paddle.nn.Layer](https://pytorch.org/docs/stable/nn.html#paddle.nn.Layer) subclass.
    Use it as a regular Paddle Layer and refer to the Paddle documentation for all matter related to general usage
    and behavior.


    Parameters:
"""


# Copied from transformers.models.deberta.modeling_deberta.DebertaModel with Deberta->DebertaV2
class DebertaV2Model(DebertaV2PreTrainedModel):
    def __init__(self,
                 _name_or_path="cache/deberta-v-xxlarge",
                 attention_head_size=64,
                 attention_probs_dropout_prob=0.1,
                 conv_act="gelu",
                 conv_kernel_size=3,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 hidden_size=1536,
                 initializer_range=0.02,
                 intermediate_size=6144,
                 layer_norm_eps=1e-07,
                 max_position_embeddings=512,
                 max_relative_positions=-1,
                 model_type="deberta-v2",
                 norm_rel_ebd="layer_norm",
                 num_attention_heads=24,
                 num_hidden_layers=48,
                 pad_token_id=0,
                 pooler_dropout=0,
                 pooler_hidden_act="gelu",
                 pooler_hidden_size=1536,
                 pos_att_type=["p2c", "c2p"],
                 position_biased_input=False,
                 position_buckets=256,
                 relative_attention=True,
                 share_att_key=True,
                 type_vocab_size=0,
                 vocab_size=128100,
                 output_attentions=False,
                 output_hidden_states=False,
                 use_return_dict=True):
        super().__init__()

        self.embeddings = DebertaV2Embeddings(
            max_position_embeddings=max_position_embeddings,
            position_biased_input=position_biased_input,
            pad_token_id=pad_token_id,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps)
        self.encoder = DebertaV2Encoder(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            attention_head_size=attention_head_size,
            relative_attention=relative_attention,
            max_relative_positions=max_relative_positions,
            max_position_embeddings=max_position_embeddings,
            position_buckets=position_buckets,
            hidden_size=hidden_size,
            norm_rel_ebd=norm_rel_ebd,
            conv_kernel_size=conv_kernel_size,
            hidden_act=hidden_act,
            conv_act=conv_act,
            intermediate_size=intermediate_size,
            share_att_key=share_att_key,
            pos_att_type=pos_att_type,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob, )
        self.z_steps = 0
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError(
            "The prune function is not implemented in DeBERTa model.")

    def forward(
            self,
            input_ids: Optional[paddle.Tensor]=None,
            attention_mask: Optional[paddle.Tensor]=None,
            token_type_ids: Optional[paddle.Tensor]=None,
            position_ids: Optional[paddle.Tensor]=None,
            inputs_embeds: Optional[paddle.Tensor]=None,
            output_attentions: Optional[bool]=None,
            output_hidden_states: Optional[bool]=None,
            return_dict: Optional[bool]=None, ) -> Union[Tuple,
                                                         BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds, )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict, )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings, )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,
                    ) + encoder_outputs[(1 if output_hidden_states else 2):]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states
            if output_hidden_states else None,
            attentions=encoder_outputs.attentions, )


def get_debertav2_model(name, pretrained=True):
    if name is None:
        return None
    model = DebertaV2Model(
        _name_or_path=name,
        attention_head_size=64,
        attention_probs_dropout_prob=0.1,
        conv_act="gelu",
        conv_kernel_size=3,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=1536,
        initializer_range=0.02,
        intermediate_size=6144,
        layer_norm_eps=1e-07,
        max_position_embeddings=512,
        max_relative_positions=-1,
        model_type="deberta-v2",
        norm_rel_ebd="layer_norm",
        num_attention_heads=24,
        num_hidden_layers=48,
        pad_token_id=0,
        pooler_dropout=0,
        pooler_hidden_act="gelu",
        pooler_hidden_size=1536,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
        position_buckets=256,
        relative_attention=True,
        share_att_key=True,
        type_vocab_size=0,
        vocab_size=128100,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True, )
    if pretrained:
        checkpoint = paddle.load(name + '/debertav2.pd', return_numpy=True)
        model.set_state_dict(checkpoint['model'])
    model.eval()
    for p in model.parameters():
        p.stop_gradient = True

    return model


def dict_from_json_file(name):
    with open(name + '/config.json', "r", encoding="utf-8") as reader:
        text = reader.read()
        config_dict = json.loads(text)
        return config_dict


def debertav2_encode_text(debertav2, texts, tokenizer, return_attn_mask=False):
    token_ids, attn_mask = debertav2_tokenize(texts, tokenizer)
    debertav2.eval()
    with paddle.no_grad():
        output = debertav2(input_ids=token_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()
    attn_mask = attn_mask.cast(bool)
    encoded_text = paddle.where(attn_mask[:, :, None] == 0,
                                paddle.to_tensor(0.), encoded_text)

    if return_attn_mask:
        return encoded_text, attn_mask

    return encoded_text


def get_debertav2_encoded_dim(name):
    return dict_from_json_file(name)['hidden_size']


if __name__ == '__main__':
    model = get_debertav2_model(
        name='/dbq/codes/CL/paddle-imagen/cache/deberta-v-xxlarge',
        pretrained=False)
