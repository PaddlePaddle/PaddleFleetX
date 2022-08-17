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

import collections
import logging

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.fluid import layers
from paddle.nn.layer.transformer import _convert_param_attr_to_list
import paddle.incubate as incubate

from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer, SharedLayerDesc
from paddle.distributed.fleet.utils import recompute
import sys
from .config import configurable


def parallel_matmul(lm_output, logit_weights, parallel_output):
    """
    """
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_parallel_group)

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None,
                 fuse=True,
                 num_partitions=1,
                 fused_linear=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights
        self.fuse = fuse

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        assert self.num_heads % num_partitions == 0, "num_heads {} must be divisible by num_partitions {}".format(
            self.num_heads, num_partitions)
        self.num_heads = self.num_heads // num_partitions

        if self.fuse:
            assert self.kdim == embed_dim
            assert self.vdim == embed_dim

            self.qkv_proj = fleet.meta_parallel.ColumnParallelLinear(
                embed_dim,
                3 * embed_dim,
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False,
                fuse_matmul_bias=fused_linear)
        else:
            self.q_proj = fleet.meta_parallel.ColumnParallelLinear(
                embed_dim,
                embed_dim,
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False,
                fuse_matmul_bias=fused_linear)

            self.k_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.kdim,
                embed_dim,
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False,
                fuse_matmul_bias=fused_linear)

            self.v_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.vdim,
                embed_dim,
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False,
                fuse_matmul_bias=fused_linear)

        self.out_proj = fleet.meta_parallel.RowParallelLinear(
            embed_dim,
            embed_dim,
            weight_attr=weight_attr,
            has_bias=True,
            input_is_parallel=True,
            fuse_matmul_bias=fused_linear)

    def _fuse_prepare_qkv(self, query):
        mix_layer = self.qkv_proj(query)
        mix_layer = paddle.reshape_(mix_layer,
                                    [0, 0, self.num_heads, 3 * self.head_dim])
        mix_layer = paddle.transpose(mix_layer, [0, 2, 1, 3])
        q, k, v = paddle.split(mix_layer, num_or_sections=3, axis=-1)
        return q, k, v

    def _prepare_qkv(self, query, key, value, use_cache=False, cache=None):
        r"""
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.

        """
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
        if use_cache is True:
            cache = self.Cache(k, v)

        return (q, k, v) if use_cache is False else (q, k, v, cache)

    def compute_kv(self, key, value):
        r"""
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.

        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.

        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        """
        Generates cache for `forward` usage in inference accroding to arguments.
        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            v = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self,
                query,
                key,
                value,
                attn_mask=None,
                use_cache=False,
                cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if use_cache is False:
            if self.fuse:
                q, k, v = self._fuse_prepare_qkv(query)
            else:
                q, k, v = self._prepare_qkv(query, key, value, use_cache,
                                            cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache,
                                               cache)
        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)

        # if attn_mask is not None:
        #     product = product + attn_mask

        # weights = F.softmax(product)

        weights = incubate.softmax_mask_fuse_upper_triangle(product)

        if self.dropout:
            with get_rng_state_tracker().rng_state('local_seed'):
                weights = F.dropout(
                    weights,
                    self.dropout,
                    training=self.training,
                    mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if use_cache:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerDecoder(nn.Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(self,
                 decoder_layers,
                 num_layers,
                 norm=None,
                 hidden_size=None,
                 use_recompute=False,
                 recompute_granularity="full"):
        super(TransformerDecoder, self).__init__()

        self.num_layers = num_layers
        self.layers = decoder_layers
        self.norm = norm
        self.use_recompute = use_recompute
        self.recompute_granularity = recompute_granularity
        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(hidden_size, epsilon=1e-5)
        elif norm is not None:
            raise ValueError("Only support LayerNorm")

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                use_cache=False,
                cache=None):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = []

        for i, mod in enumerate(self.layers):
            if cache is None:
                if use_cache:
                    output, new_cache = mod(output,
                                            memory,
                                            tgt_mask=tgt_mask,
                                            use_cache=use_cache,
                                            cache=cache)
                    new_caches.append(new_cache)
                else:
                    if self.use_recompute and self.recompute_granularity == "full":
                        output = recompute(mod, output, memory, tgt_mask,
                                           use_cache, cache)
                    else:
                        output = mod(output, memory, tgt_mask, use_cache,
                                     cache)

            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        use_cache=use_cache,
                                        cache=cache[i])
                new_caches.append(new_cache)

        if self.norm is not None:
            output = self.norm(output)
        return output if use_cache is False else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        r"""
        Generates cache for `forward` usage. The generated cache is a list, and
        each element in it is a tuple( :code:`(incremental_cache, static_cache)` )
        produced by `TransformerDecoderLayer.gen_cache`. See `TransformerDecoderLayer.gen_cache`
        for more details. If `do_zip` is True, apply `zip` on these tuples to get
        a list with two elements.
       """
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache


class TransformerDecoderLayer(nn.Layer):
    """
    The transformer decoder layer.

    It contains multiheadattention and some linear layers.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="gelu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=True,
                 weight_attr=None,
                 bias_attr=None,
                 num_partitions=1,
                 fused_linear=False,
                 recompute_attn=False):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before
        self.recompute_attn = recompute_attn

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            num_partitions=num_partitions,
            fused_linear=fused_linear)

        self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            d_model,
            dim_feedforward,
            weight_attr=weight_attrs[2],
            gather_output=False,
            has_bias=True,
            fuse_matmul_bias=fused_linear)

        self.linear2 = fleet.meta_parallel.RowParallelLinear(
            dim_feedforward,
            d_model,
            weight_attr=weight_attrs[2],
            input_is_parallel=True,
            has_bias=True,
            fuse_matmul_bias=fused_linear)

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self,
                tgt,
                memory=None,
                tgt_mask=None,
                use_cache=False,
                cache=None):
        residual = tgt

        if self.normalize_before:
            tgt = self.norm1(tgt)

        if use_cache is False:
            if self.recompute_attn:
                tgt = recompute(self.self_attn, tgt, None, None, tgt_mask,
                                use_cache, cache)
            else:
                tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    use_cache, cache)

        with get_rng_state_tracker().rng_state('global_seed'):
            tgt = residual + self.dropout1(tgt)

        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)

        with get_rng_state_tracker().rng_state('global_seed'):
            tgt = self.dropout2(
                self.linear2(F.gelu(
                    self.linear1(tgt), approximate=True)))

        tgt = residual + tgt

        if not self.normalize_before:
            tgt = self.norm2(tgt)

        return tgt if use_cache is False else (tgt, incremental_cache)

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(
            memory, type=self.self_attn.Cache)
        return incremental_cache


class GPTEmbeddings(nn.Layer):
    """
    Include embeddings from word and position embeddings.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        super(GPTEmbeddings, self).__init__()

        self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                mean=0.0, std=initializer_range)))

        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                mean=0.0, std=initializer_range)))

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embedings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class GPTModel(nn.Layer):
    @configurable
    def __init__(self,
                 vocab_size=51200,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 ffn_hidden_size=3072,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 num_partitions=1,
                 use_recompute=False,
                 fused_linear=False,
                 recompute_granularity="full"):

        super(GPTModel, self).__init__()

        recompute_attn = use_recompute and recompute_granularity == "only_attn"

        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embeddings = GPTEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, self.initializer_range)

        decoder_layers = nn.LayerList()
        for i in range(num_layers):
            decoder_layers.append(
                TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=ffn_hidden_size,
                    dropout=hidden_dropout_prob,
                    activation="gelu",
                    attn_dropout=attention_probs_dropout_prob,
                    act_dropout=hidden_dropout_prob,
                    weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(
                            mean=0.0, std=self.initializer_range)),
                    bias_attr=None,
                    num_partitions=num_partitions,
                    fused_linear=fused_linear,
                    recompute_attn=recompute_attn))

        self.decoder = TransformerDecoder(
            decoder_layers,
            num_layers,
            norm="LayerNorm",
            hidden_size=hidden_size,
            use_recompute=use_recompute,
            recompute_granularity=recompute_granularity)

    @classmethod
    def from_config(cls, cfg):
        return {
            "vocab_size": cfg['vocab_size'],
            "hidden_size": cfg['hidden_size'],
            "num_layers": cfg['num_layers'],
            "num_attention_heads": cfg['num_attention_heads'],
            "ffn_hidden_size": cfg['ffn_hidden_size'],
            "hidden_dropout_prob": cfg['hidden_dropout_prob'],
            "attention_probs_dropout_prob":
            cfg['attention_probs_dropout_prob'],
            "max_position_embeddings": cfg['max_position_embeddings'],
            "type_vocab_size": cfg['type_vocab_size'],
            "initializer_range": cfg['initializer_range'],
            "num_partitions": cfg['topology'].get_dim('model'),
            "use_recompute": cfg['use_recompute'],
            "fused_linear": cfg['fused_linear'],
            "recompute_granularity": cfg['recompute_granularity']
        }

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                cache=None):

        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = paddle.shape(cache[0].k)[-2]
            position_ids = paddle.arange(
                past_length,
                paddle.shape(input_ids)[-1] + past_length,
                dtype=input_ids.dtype)
            position_ids = position_ids.unsqueeze(0)
            # .expand_as(input_ids)
            position_ids = paddle.fluid.layers.expand_as(position_ids,
                                                         input_ids)
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)

        encoder_outputs = self.decoder(
            embedding_output,
            memory=None,
            # tgt_mask=attention_mask,
            tgt_mask=None,  # use softmax_mask_fuse_upper_triangle
            use_cache=use_cache,
            cache=cache)

        return encoder_outputs


class GPTForPretraining(nn.Layer):
    """
    GPT Model with pretraining tasks on top.

    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.

    """

    def __init__(self, gpt):
        super(GPTForPretraining, self).__init__()
        self.gpt = gpt
        # extra_parameters using for sharding stage3 to register extra_parameters
        self.extra_parameters = [self.gpt.embeddings.word_embeddings.weight]

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                masked_positions=None,
                use_cache=False,
                cache=None):

        outputs = self.gpt(input_ids,
                           position_ids=position_ids,
                           attention_mask=attention_mask,
                           use_cache=use_cache,
                           cache=cache)
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        logits = parallel_matmul(
            encoder_outputs, self.gpt.embeddings.word_embeddings.weight, True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPTPretrainingCriterion(nn.Layer):
    """
    Criterion for GPT. It calculates the final loss.
    """

    def __init__(self, topo=None):
        super(GPTPretrainingCriterion, self).__init__()
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        self.parallel_loss_func = fleet.meta_parallel.ParallelCrossEntropy()

    def forward(self, prediction_scores, masked_lm_labels, loss_mask):
        """
        Args:
            prediction_scores(Tensor):
                The logits of masked token prediction. Its data type should be float32 and
                its shape is [batch_size, sequence_length, vocab_size].
            masked_lm_labels(Tensor):
                The labels of the masked language modeling, the dimensionality of `masked_lm_labels`
                is equal to `prediction_scores`. Its data type should be int64 and
                its shape is [batch_size, sequence_length, 1].
            loss_mask(Tensor):
                Mask used for calculating the loss of the masked language modeling to avoid
                calculating some unwanted tokens.
                Its data type should be float32 and its shape is [batch_size, sequence_length, 1].

        Returns:
            Tensor: The pretraining loss. Its data type should be float32 and its shape is [1].

        """
        hcg = fleet.get_hybrid_communicate_group()
        mp_size = hcg.get_model_parallel_world_size()
        if mp_size > 1:
            masked_lm_loss = self.parallel_loss_func(
                prediction_scores, masked_lm_labels.unsqueeze(2))
        else:
            masked_lm_loss = self.loss_func(prediction_scores,
                                            masked_lm_labels.unsqueeze(2))

        loss_mask = loss_mask.reshape([-1])
        masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
        loss = masked_lm_loss / loss_mask.sum()
        return loss


# these Layers is just for PipelineParallel


class GPTPretrainingCriterionPipe(GPTPretrainingCriterion):
    """Extends GPTPretrainingCriterion to meet the input standard."""

    def forward(self, prediction_scores, args):
        masked_lm_labels = args[0]
        loss_mask = args[1]
        loss = super().forward(prediction_scores, masked_lm_labels, loss_mask)
        return loss


class EmbeddingPipe(GPTEmbeddings):
    """Extends GPTEmbeddings to forward attention_mask through the pipeline."""

    @property
    def embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, tensors):
        input_ids, position_ids = tensors
        embeddings = super().forward(
            input_ids=input_ids, position_ids=position_ids)
        return embeddings


class GPTForPretrainingPipe(PipelineLayer):
    """GPTForPretraining adapted for pipeline parallelism.

    The largest change is flattening the GPTModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    @configurable
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 ffn_hidden_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 num_partitions=1,
                 topology=None,
                 use_recompute=False,
                 fused_linear=False,
                 recompute_granularity="full"):

        recompute_attn = use_recompute and recompute_granularity == "only_attn"

        # forward desc
        self.descs = []

        self.descs.append(
            SharedLayerDesc(
                'embed',
                EmbeddingPipe,
                shared_weight_attr='embedding_weight',
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                hidden_dropout_prob=hidden_dropout_prob,
                max_position_embeddings=max_position_embeddings,
                type_vocab_size=type_vocab_size,
                initializer_range=0.02))

        for _ in range(num_layers):
            self.descs.append(
                LayerDesc(
                    TransformerDecoderLayer,
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=ffn_hidden_size,
                    dropout=hidden_dropout_prob,
                    activation=hidden_act,
                    attn_dropout=attention_probs_dropout_prob,
                    act_dropout=hidden_dropout_prob,
                    weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(
                            mean=0.0, std=initializer_range)),
                    bias_attr=None,
                    num_partitions=num_partitions,
                    fused_linear=fused_linear,
                    recompute_attn=recompute_attn))

        self.descs.append(
            LayerDesc(
                nn.LayerNorm, normalized_shape=hidden_size))

        def _logits_helper(embedding, output):
            return parallel_matmul(output, embedding.embedding_weight, True)

        self.descs.append(
            SharedLayerDesc(
                'embed',
                EmbeddingPipe,
                forward_func=_logits_helper,
                shared_weight_attr='embedding_weight',
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                hidden_dropout_prob=hidden_dropout_prob,
                max_position_embeddings=max_position_embeddings,
                type_vocab_size=type_vocab_size,
                initializer_range=0.02))

        recompute_interval = 0
        if recompute and not recompute_attn:
            recompute_interval = 1

        super().__init__(
            layers=self.descs,
            loss_fn=GPTPretrainingCriterionPipe(),
            topology=topology,
            seg_method="layer:TransformerDecoderLayer",
            recompute_interval=recompute_interval,
            recompute_partition=False,
            recompute_offload=False)

    @classmethod
    def from_config(cls, cfg):
        return {
            "vocab_size": cfg['vocab_size'],
            "hidden_size": cfg['hidden_size'],
            "num_layers": cfg['num_layers'],
            "num_attention_heads": cfg['num_attention_heads'],
            "ffn_hidden_size": cfg['ffn_hidden_size'],
            "hidden_dropout_prob": cfg['hidden_dropout_prob'],
            "attention_probs_dropout_prob":
            cfg['attention_probs_dropout_prob'],
            "max_position_embeddings": cfg['max_position_embeddings'],
            "type_vocab_size": cfg['type_vocab_size'],
            "initializer_range": cfg['initializer_range'],
            "num_partitions": cfg['topology'].get_dim('model'),
            "use_recompute": cfg['use_recompute'],
            "topology": cfg['topology'],
            "fused_linear": cfg['fused_linear'],
            "recompute_granularity": cfg['recompute_granularity']
        }
