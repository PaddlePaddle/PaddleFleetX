# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import collections
import math
import unittest
import random
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
import paddle.utils as utils
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.framework import in_dygraph_mode
from paddle.nn.layer.transformer import _convert_param_attr_to_list
from paddle.fluid.initializer import Normal, Constant, NumpyArrayInitializer
import paddle.static as static
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.context import get_default_distributed_context
from paddle.distributed import fleet
from args import parse_args
import paddlenlp
import global_setting

paddle.enable_static()


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
                 topo=None,
                 fuse=False,
                 debug=False):
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

        if topo is None or topo.mp_info.size == 1:
            if self.fuse:
                assert self.kdim == embed_dim
                assert self.vdim == embed_dim
                self.qkv_proj = nn.Linear(
                    embed_dim, 3 * embed_dim, weight_attr, bias_attr=bias_attr)
            else:
                if debug:
                    np.random.seed(2021)
                    arr = np.random.normal(0, 0.02, size=(embed_dim, embed_dim))
                    weight_attr = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr))
                else:
                    weight_attr = weight_attr
                self.q_proj = nn.Linear(
                    embed_dim, embed_dim, weight_attr=weight_attr, bias_attr=bias_attr)
                self.k_proj = nn.Linear(
                    self.kdim, embed_dim, weight_attr=weight_attr, bias_attr=bias_attr)
                self.v_proj = nn.Linear(
                    self.vdim, embed_dim, weight_attr=weight_attr, bias_attr=bias_attr)

            if debug:
                np.random.seed(2021)
                arr = np.random.normal(0, 0.02, size=(embed_dim, embed_dim))
                weight_attr = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr))
            else:
                weight_attr = weight_attr
            self.out_proj = nn.Linear(
                embed_dim, embed_dim, weight_attr=weight_attr, bias_attr=bias_attr)
        else:
            assert self.num_heads % topo.mp_info.size == 0
            self.num_heads = self.num_heads // topo.mp_info.size
            if self.fuse:
                assert self.kdim == embed_dim
                assert self.vdim == embed_dim
                self.qkv_proj = paddlenlp.ops.ColumnParallelLiner(
                    (embed_dim, 3 * embed_dim),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=weight_attr,
                    bias_attr=bias_attr)
            else:
                if debug:
                    np.random.seed(2021)
                    arr = np.random.normal(0, 0.02, size=(embed_dim, embed_dim))
                    rank = fleet.worker_index()
                    inner_rank = rank % topo.mp_info.size
                    start_index = inner_rank * (embed_dim // topo.mp_info.size)
                    end_index = start_index + (embed_dim // topo.mp_info.size)
                    arr_column = arr[:, start_index:end_index]
                    arr_row = arr[start_index:end_index, :]
                    weight_attr1 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr_column))
                    weight_attr2 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr_row))
                else:
                    weight_attr1 = weight_attr
                    weight_attr2 = weight_attr


                self.q_proj = paddlenlp.ops.ColumnParallelLiner(
                    (embed_dim, embed_dim),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=weight_attr1,
                    bias_attr=bias_attr)
                self.k_proj = paddlenlp.ops.ColumnParallelLiner(
                    (self.kdim, embed_dim),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=weight_attr1,
                    bias_attr=bias_attr)
                self.v_proj = paddlenlp.ops.ColumnParallelLiner(
                    (self.vdim, embed_dim),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=weight_attr1,
                    bias_attr=bias_attr)

            self.out_proj = paddlenlp.ops.RowParallelLiner(
                (embed_dim, embed_dim),
                topo.mp_info.size,
                input_is_parallel=True,
                param_attr=weight_attr2,
                bias_attr=bias_attr)

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

        if global_setting._global_parallel_stratergy == "mp":
            auto.shard_tensor(
                self.q_proj.weight, global_setting._global_process_mesh, dim_mapping=[-1, 0])
        elif global_setting._global_parallel_stratergy == "dp_mp":
            auto.shard_tensor(
                self.q_proj.weight, global_setting._global_process_mesh, dim_mapping=[-1, 1])

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

        if global_setting._global_parallel_stratergy == "mp":
            auto.shard_tensor(
                self.k_proj.weight, global_setting._global_process_mesh, dim_mapping=[-1, 0])
        elif global_setting._global_parallel_stratergy == "dp_mp":
            auto.shard_tensor(
                self.k_proj.weight, global_setting._global_process_mesh, dim_mapping=[-1, 1])

        v = self.v_proj(value)

        if global_setting._global_parallel_stratergy == "mp":
            auto.shard_tensor(
                self.v_proj.weight, global_setting._global_process_mesh, dim_mapping=[-1, 0])
        elif global_setting._global_parallel_stratergy == "dp_mp":
            auto.shard_tensor(
                self.v_proj.weight, global_setting._global_process_mesh, dim_mapping=[-1, 1])

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
                q, k, v = self._prepare_qkv(query, key, value, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache,
                                               cache)

        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)

        if attn_mask is not None:
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
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
        if global_setting._global_parallel_stratergy == "mp":
            auto.shard_tensor(
                self.out_proj.weight,
                global_setting._global_process_mesh,
                dim_mapping=[0, -1])
        elif global_setting._global_parallel_stratergy == "dp_mp":
            auto.shard_tensor(
                self.out_proj.weight,
                global_setting._global_process_mesh,
                dim_mapping=[1, -1])

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
                 topo=None,
                 debug=False):
        super(TransformerDecoder, self).__init__()

        self.topo = topo
        self.num_layers = num_layers
        self.layers = decoder_layers
        self.norm = norm
        if norm is "LayerNorm":
            self.norm = nn.LayerNorm(hidden_size)
        elif norm is not None:
            raise ValueError("Only support LayerNorm")
        self.block = paddle.static.default_main_program().global_block()
        self.checkpoints = []
        self.debug = debug

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
        self.checkpoints = []

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
                    output = mod(output,
                                 memory,
                                 tgt_mask=tgt_mask,
                                 use_cache=use_cache,
                                 cache=cache)

            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        use_cache=use_cache,
                                        cache=cache[i])
                new_caches.append(new_cache)
            self.checkpoints.append(output.name)

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
                 topo=None,
                 debug=False):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3
        self.block = paddle.static.default_main_program().global_block()


        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            topo=topo,
            debug=debug)
        # fuse elementwise_add gelu
        if topo is None or topo.mp_info.size == 1:
            if debug:
                np.random.seed(2021)
                arr1 = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
                arr2 = np.random.normal(0, 0.02, size=(dim_feedforward, d_model))
                weight_attr1 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr1))
                weight_attr2 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr2))
            else:
                weight_attr1 = weight_attrs[2]
                weight_attr2 = weight_attrs[2]
                
            self.linear1 = nn.Linear(
                d_model,
                dim_feedforward,
                weight_attr1,
                bias_attr=bias_attrs[2])
            self.linear2 = nn.Linear(
                dim_feedforward,
                d_model,
                weight_attr2,
                bias_attr=bias_attrs[2])
            
        else:
            if debug:
                np.random.seed(2021)
                arr1 = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
                arr2 = np.random.normal(0, 0.02, size=(dim_feedforward, d_model))
                rank = fleet.worker_index()
                inner_rank = rank % topo.mp_info.size
                start_index = inner_rank * (dim_feedforward // topo.mp_info.size)
                end_index = start_index + (dim_feedforward // topo.mp_info.size)
                arr1_column = arr1[:, start_index:end_index]
                arr2_row = arr2[start_index:end_index, :]
                weight_attr1 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr1_column))
                weight_attr2 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr2_row))
            else:
                weight_attr1 = weight_attrs[2]
                weight_attr2 = weight_attrs[2]
            self.linear1 = paddlenlp.ops.ColumnParallelLiner(
                (d_model, dim_feedforward),
                topo.mp_info.size,
                gather_out=False,
                param_attr=weight_attr1,
                bias_attr=bias_attrs[2])
            self.linear2 = paddlenlp.ops.RowParallelLiner(
                (dim_feedforward, d_model),
                topo.mp_info.size,
                input_is_parallel=True,
                param_attr=weight_attr2,
                bias_attr=bias_attrs[2])

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, use_cache=False, cache=None):
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if use_cache is False:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    use_cache, cache)

        tgt = residual + self.dropout1(tgt)

        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)

        if global_setting._global_parallel_stratergy == "mp":
            auto.shard_tensor(
                self.linear1.weight, global_setting._global_process_mesh,
                dim_mapping=[-1, 0])
        elif global_setting._global_parallel_stratergy == "dp_mp":
            auto.shard_tensor(
                self.linear1.weight, global_setting._global_process_mesh,
                dim_mapping=[-1, 1])

        if global_setting._global_parallel_stratergy == "mp":
            auto.shard_tensor(
                self.linear2.weight, global_setting._global_process_mesh,
                dim_mapping=[0, -1])
        elif global_setting._global_parallel_stratergy == "dp_mp":
            auto.shard_tensor(
                self.linear2.weight, global_setting._global_process_mesh,
                dim_mapping=[1, -1])

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
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 topo=None,
                 debug=False):
        super(GPTEmbeddings, self).__init__()
        if topo is None or topo.mp_info.size == 1:
            if debug:
                np.random.seed(2021)
                arr = np.random.normal(0, 0.02, size=(50304, 768))
                weight_attr = paddle.ParamAttr(name="word_embeddings",
                                initializer=NumpyArrayInitializer(arr))  
            else:
                weight_attr = paddle.ParamAttr(name="word_embeddings",
                                initializer=nn.initializer.Normal(
                                mean=0.0, std=initializer_range))
            self.word_embeddings = nn.Embedding(
                vocab_size,
                hidden_size,
                weight_attr=weight_attr)
        else:
            if debug:
                np.random.seed(2021)
                arr = np.random.normal(0, 0.02, size=(50304, 768))
                rank = fleet.worker_index()
                inner_rank = rank % topo.mp_info.size
                start_index = inner_rank * (50304 // topo.mp_info.size)
                end_index = start_index + (50304 // topo.mp_info.size)
                arr_row = arr[start_index:end_index, :]
                weight_attr = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr_row))
            else:
                weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range))


            self.word_embeddings = paddlenlp.ops.ParallelEmbedding(
                vocab_size,
                hidden_size,
                topo,
                weight_attr=weight_attr)

        pos_emb = np.random.normal(loc = 0, scale = 0.02, size = (max_position_embeddings, hidden_size)) 
        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                name="pos_embeddings",
                initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)))

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones

        input_embedings = self.word_embeddings(input_ids)

        if global_setting._global_parallel_stratergy == "mp":
            auto.shard_tensor(
                self.word_embeddings.weight,
                global_setting._global_process_mesh,
                dim_mapping=[0, -1])
        elif global_setting._global_parallel_stratergy == "dp_mp":
            auto.shard_tensor(
                self.word_embeddings.weight,
                global_setting._global_process_mesh,
                dim_mapping=[1, -1])

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embedings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class GPTModel(nn.Layer):
    """
    The base model of gpt.
    """

    def __init__(self,
                 vocab_size=50304,
                 hidden_size=768,
                 num_hidden_layers=2,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 eos_token_id=7,
                 bos_token_id=0,
                 eol_token_id=3,
                 topo=None,
                 debug=False):
        super(GPTModel, self).__init__()

        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.topo = topo
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.block = paddle.static.default_main_program().global_block()


        self.pipline_mode = topo is not None and topo.pp_info.size > 1
        if self.pipline_mode:
            self.layer_per_stage = num_hidden_layers // self.topo.pp_info.size

        self.embeddings = GPTEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, self.initializer_range,
            topo, debug=debug)

        decoder_layers = nn.LayerList()
        for i in range(num_hidden_layers):
            DecoderLayer = TransformerDecoderLayer
            if self.pipline_mode:
                DecoderLayer = paddlenlp.ops.guard('gpu:{}'.format(
                    i // self.layer_per_stage))(TransformerDecoderLayer)
            decoder_layers.append(
                DecoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=intermediate_size,
                    dropout=hidden_dropout_prob,
                    activation=hidden_act,
                    attn_dropout=attention_probs_dropout_prob,
                    act_dropout=hidden_dropout_prob,
                    weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(
                            mean=0.0, std=self.initializer_range)),
                    bias_attr=None,
                    topo=topo, debug=debug))

        if self.pipline_mode:
            Decoder = paddlenlp.ops.guard('gpu:{}'.format(
                self.topo.pp_info.size - 1))(TransformerDecoder)
        else:
            Decoder = TransformerDecoder

        self.decoder = Decoder(
            decoder_layers,
            num_hidden_layers,
            norm="LayerNorm",
            hidden_size=hidden_size,
            topo=topo,
            debug=debug)

        self.checkpoints = []

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                cache=None):
        self.checkpoints = []
        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = paddle.shape(cache[0].k)[-2]
            position_ids = paddle.arange(
                past_length,
                paddle.shape(input_ids)[-1] + past_length,
                dtype='int64')
            position_ids = position_ids.unsqueeze(0)
            # .expand_as(input_ids)
            position_ids = paddle.fluid.layers.expand_as(position_ids,
                                                         input_ids)
        print("input_ids.name", input_ids.name)
        print("has var: ", self.block.has_var(input_ids.name))
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)

        attention_mask.stop_gradient = True

        encoder_outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=attention_mask,
            use_cache=use_cache,
            cache=cache)
        self.checkpoints.extend(self.decoder.checkpoints)
        return encoder_outputs


class GPTForPretraining(nn.Layer):
    """
    The pretraining model of GPT.
    It returns some logits and cached_kvs.
    """

    def __init__(
            self,
            gpt,
            vocab_size=50304,
            hidden_size=768,
            initializer_range=0.02, ):
        super(GPTForPretraining, self).__init__()
        final_emb = np.random.normal(loc = 0, scale = 0.02, size = (vocab_size, hidden_size))
        self.output_embeddings = nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                name="output_embeddings",
                initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)))

        self.gpt = gpt

    def parallel_matmul(self, lm_output, logit_weights, parallel_output, topo):
        if topo is not None and topo.mp_info.size > 1:
            input_parallel = paddle.distributed.collective._c_identity(
                lm_output, group=None)

            logits = paddle.matmul(
                input_parallel, logit_weights, transpose_y=True)

            if parallel_output:
                return logits

            return paddle.distributed.collective._c_concat(logits, group=None)
        else:
            logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
            return logits

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
        logits = self.parallel_matmul(
            encoder_outputs, self.output_embeddings.weight, True, None)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPTPretrainingCriterion(nn.Layer):
    """
    Criterion for GPT.
    It calculates the final loss.
    """

    def __init__(self, args=None, topo=None):
        super(GPTPretrainingCriterion, self).__init__()
        self.block = paddle.static.default_main_program().global_block()
        self.topo = topo
        self.args = args
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")

    def forward(self, prediction_scores, masked_lm_labels, loss_mask):
        masked_lm_loss = self.loss_func(prediction_scores,
                                        masked_lm_labels.unsqueeze(2))
        loss_mask = loss_mask.reshape([-1])
        masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
        total_loss = masked_lm_loss / loss_mask.sum()
        pp_total_loss = None
        if self.topo.pp_info.size > 1:
            total_loss = total_loss
            masked_lm_loss.persistable = True
            total_loss.persistable = True
            total_loss.persistable = True
            pp_total_loss = paddle.fluid.layers.fill_constant([1, ], "float32", 0.0)
            pp_total_loss.persistable = True
            block = paddle.static.default_main_program().global_block()
            acc_steps = self.args.global_batch_size // self.topo.data_info.size // self.args.micro_batch_size
            tmp = total_loss / acc_steps
            block.append_op(
                type="elementwise_add",
                inputs={
                    "X": [pp_total_loss],
                    "Y": [tmp]
                },
                outputs={"Out": [pp_total_loss]}
            )
        loss_vars = {
            "total_loss": total_loss,
            "pp_total_loss": pp_total_loss
        }
        return loss_vars

