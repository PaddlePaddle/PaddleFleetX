# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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

import math
import numpy as np

import paddle
import paddle.fluid.layers as layers
from paddle import nn

from .utils import (
    ACT2FN, find_pruneable_heads_and_indices, prune_linear_layer,
    BaseModelOutputWithPastAndCrossAttentions)

__all__ = ['get_t5_model', 't5_encode_text', 'get_encoded_dim']


def finfo(dtype):
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


class T5LayerNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = self.create_parameter(
            [hidden_size], default_initializer=nn.initializer.Constant(value=1.))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        # variance = hidden_states.cast(paddle.float32).pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)

        # hidden_32 = hidden_states.cast(paddle.float32)
        # power = paddle.to_tensor([2], dtype='float32')
        # pow_data = layers.elementwise_pow(hidden_32, power)
        # variance = pow_data.mean(-1, keepdim=True)
        # var_add_epi = variance + self.variance_epsilon
        # var_add_epi_sqrt = paddle.sqrt(var_add_epi)
        # hidden_states = layers.elementwise_div(hidden_states, var_add_epi_sqrt)

        # convert into half-precision if necessary
        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = hidden_states.cast(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseActDense(nn.Layer):
    def __init__(self, d_model, d_ff, dropout_rate, dense_act_fn):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias_attr=False)
        self.wo = nn.Linear(d_ff, d_model, bias_attr=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = ACT2FN[dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Layer):
    def __init__(self, d_model, d_ff, dropout_rate, dense_act_fn):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias_attr=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias_attr=False)
        self.wo = nn.Linear(d_ff, d_model, bias_attr=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = ACT2FN[dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

    
class T5LayerFF(nn.Layer):
    def __init__(self, 
                 d_model,
                 d_ff,
                 dropout_rate,
                 layer_norm_epsilon,
                 feed_forward_proj):
        super().__init__()
        if feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedActDense(
                d_model, d_ff, dropout_rate, dense_act_fn)
        elif feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseActDense(
                d_model, d_ff, dropout_rate, feed_forward_proj)

        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Layer):
    def __init__(self, 
                 is_decoder,
                 relative_attention_num_buckets,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.d_model = d_model
        self.key_value_proj_dim = d_kv
        self.n_heads = num_heads
        self.dropout = dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias_attr=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).cast(paddle.int64) * num_buckets
            relative_position = paddle.abs(relative_position)
        else:
            relative_position = -paddle.min(relative_position, paddle.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            paddle.log(relative_position.cast('float32') / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).cast(paddle.int64)
        relative_position_if_large = paddle.minimum(
            relative_position_if_large, paddle.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += paddle.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        context_position = paddle.arange(query_length, dtype=paddle.int64)[:, None]
        memory_position = paddle.arange(key_length, dtype=paddle.int64)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.transpose([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.reshape([0, -1, self.n_heads, self.key_value_proj_dim]).transpose([0, 2, 1, 3])

        def unshape(states):
            """reshape"""
            return states.transpose([0, 2, 1, 3]).reshape([batch_size, -1, self.inner_dim])

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = paddle.concat([past_key_value, hidden_states], axis=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = paddle.matmul(
            query_states, key_states.transpose([0, 1, 3, 2])
        )

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = paddle.zeros(
                    (1, self.n_heads, real_seq_length, key_length), dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.cast('float32'), axis=-1).astype(
            scores.dtype
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(paddle.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Layer):
    def __init__(self,
                 is_decoder,
                 relative_attention_num_buckets,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 layer_norm_epsilon,
                 has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(
            is_decoder,
            relative_attention_num_buckets,
            d_model,
            d_kv,
            num_heads,
            dropout_rate,
            has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Layer):
    def __init__(self, is_decoder, relative_attention_num_buckets, d_model,
                 d_kv, num_heads, dropout_rate, layer_norm_epsilon):
        super().__init__()
        self.EncDecAttention = T5Attention(is_decoder,
                                           relative_attention_num_buckets,
                                           d_model,
                                           d_kv,
                                           num_heads,
                                           has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5Block(nn.Layer):
    def __init__(self, is_decoder,
                 relative_attention_num_buckets,
                 feed_forward_proj,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 layer_norm_epsilon,
                 d_ff,
                 has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.LayerList()
        self.layer.append(T5LayerSelfAttention(
            is_decoder,
            relative_attention_num_buckets,
            d_model,
            d_kv,
            num_heads,
            dropout_rate,
            layer_norm_epsilon,
            has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(
                T5LayerCrossAttention(is_decoder,
                                      relative_attention_num_buckets,
                                      d_model,
                                      d_kv,
                                      num_heads,
                                      dropout_rate,
                                      layer_norm_epsilon))

        self.layer.append(T5LayerFF(d_model, d_ff, dropout_rate,
                                    layer_norm_epsilon, feed_forward_proj))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == paddle.float16 and paddle.isinf(hidden_states).any():
            # TODO change clamp_value
            # clamp_value = finfo(hidden_states.dtype).max - 1000
            # clamp_value = 1e10
            clamp_value = finfo(paddle.float16).max - 1000
            hidden_states = paddle.clip(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == paddle.float16 and paddle.isinf(hidden_states).any():
                # clamp_value = finfo(hidden_states.dtype).max - 1000
                # TODO change clamp_value
                # clamp_value = finfo(hidden_states.dtype).max - 1000
                clamp_value = finfo(paddle.float16).max - 1000
                hidden_states = paddle.clip(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == paddle.float16 and paddle.isinf(hidden_states).any():
            # TODO change clamp_value
            # clamp_value = finfo(hidden_states.dtype).max - 1000
            clamp_value = finfo(paddle.float16).max - 1000
            hidden_states = paddle.clip(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5Stack(nn.Layer):
    def __init__(self,
                 d_model,
                 num_layers,
                 layer_norm_epsilon,
                 dropout_rate,
                 relative_attention_num_buckets,
                 feed_forward_proj,
                 d_kv,
                 num_heads,
                 d_ff,
                 embed_tokens=None,
                 is_decoder=False):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.is_decoder = is_decoder
        self.num_layers = num_layers 

        self.block = nn.LayerList(
            [T5Block(is_decoder,
                     relative_attention_num_buckets,
                     feed_forward_proj,
                     d_model,
                     d_kv,
                     num_heads,
                     dropout_rate,
                     layer_norm_epsilon,
                     d_ff,
                     has_relative_attention_bias=bool(i == 0))
            for i in range(num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
    

    def get_extended_attention_mask(self, attention_mask, input_shape):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`paddle.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `paddle.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = extended_attention_mask.cast(dtype='float16')  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(
        self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`paddle.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `paddle.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        #head_mask = head_mask.cast(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if use_cache is True:
            assert (
                self.is_decoder
            ), f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape([-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = paddle.ones(batch_size, mask_seq_length)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = paddle.ones(
                batch_size, encoder_seq_length, dtype=paddle.int64)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

 
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)



        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5EncoderModel(nn.Layer):
    authorized_missing_keys = [
        r"encoder.embed_tokens.weight",
    ]

    def __init__(self,
                 vocab_size=32128,
                 d_model=768,
                 d_kv=64,
                 d_ff=3072,
                 num_layers=12,
                 num_decoder_layers=12,
                 num_heads=12,
                 relative_attention_num_buckets=32,
                 dropout_rate=0.1,
                 layer_norm_epsilon=1e-06,
                 feed_forward_proj="relu"
                 ):
        super().__init__()
        self.shared = nn.Embedding(vocab_size, d_model)

        use_cache = False
        is_encoder_decoder = False
        self.encoder = T5Stack(d_model,
                               num_layers,
                               layer_norm_epsilon,
                               dropout_rate,
                               relative_attention_num_buckets,
                               feed_forward_proj,
                               d_kv,
                               num_heads,
                               d_ff,
                               embed_tokens=self.shared,
                               is_decoder=False)


  
    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import T5Tokenizer, T5EncoderModel

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5EncoderModel.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else True
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

def T5Model(config):
    config = T5Config(**config)
    model = T5EncoderModel(config)
    return model

def get_t5_model(name, pretrained=True):
    model =T5EncoderModel(
                 vocab_size=32128,
                 d_model=1024,
                 d_kv=128,
                 d_ff=65536,
                #  num_layers=3,
                 num_layers=24,
                 num_decoder_layers=None,
                 num_heads=128,
                 relative_attention_num_buckets=32,
                 dropout_rate=0.,
                 layer_norm_epsilon=1e-06,
                 feed_forward_proj="relu"        
    ) 
    if pretrained:
        checkpoint = paddle.load(name +'/t5.pd', return_numpy=True) 
        model.set_state_dict(checkpoint['model'])
    return model

def t5_11b():
    return T5EncoderModel(
                 vocab_size=32128,
                 d_model=1024,
                 d_kv=128,
                 d_ff=65536,
                 num_layers=24,
                 num_decoder_layers=None,
                 num_heads=128,
                 relative_attention_num_buckets=32,
                 dropout_rate=0.,
                 layer_norm_epsilon=1e-06,
                 feed_forward_proj="relu"        
    )


def t5_encode_text(t5, token_ids, attn_mask, return_mask=True):
    t5.eval()
    with paddle.no_grad():
        output = t5(input_ids=token_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()
    if return_mask:
        attn_mask = attn_mask.cast('bool')
        return encoded_text, attn_mask
    return encoded_text


def get_encoded_dim(name):
    return dict_from_json_file(name)['d_model']