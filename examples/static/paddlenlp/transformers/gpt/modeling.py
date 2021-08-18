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

import collections
import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.fluid import layers
from paddle.fluid.framework import in_dygraph_mode
from paddle.nn.layer.transformer import _convert_param_attr_to_list
from paddle.fluid.initializer import Normal, Constant, NumpyArrayInitializer
from paddle.distributed.fleet import fleet

from .. import PretrainedModel, register_base_model
import paddlenlp

__all__ = [
    'GPTModel',
    "GPTPretrainedModel",
    'GPTForPretraining',
    'GPTPretrainingCriterion',
    'GPTForGreedyGeneration',
]


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
                 fuse=False):
        super(MultiHeadAttention, self).__init__()
        # weight_attr = paddle.ParamAttr(initializer=NumpyArrayInitializer(np.random.normal(0.0, 0.02, size=[8, 1024, embed_dim])))
        # weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.02))
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        # self.dropout = 0
        self.dropout = dropout
        self.need_weights = need_weights
        self.fuse = False

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        if topo is None or topo.mp_info.size == 1:
            if self.fuse:
                assert self.kdim == embed_dim
                assert self.vdim == embed_dim
                self.qkv_proj = nn.Linear(
                    embed_dim, 3 * embed_dim, weight_attr, bias_attr=bias_attr)
            else:
                # np.random.seed(5)
                # arr1 = np.random.normal(0, 0.02, size=(768, 384))
                # np.random.seed(5)
                # arr2 = np.random.normal(0, 0.02, size=(768, 384))
                # arr = np.concatenate([arr1, arr2], axis=-1)
                # print("single card column seed", arr[:,:384] == arr[:, 384:])
                np.random.seed(5)
                arr = np.random.normal(0, 0.02, size=(768, 768))
                self.q_proj = nn.Linear(
                    embed_dim, embed_dim, weight_attr=paddle.ParamAttr(initializer=NumpyArrayInitializer(arr)), bias_attr=bias_attr)
                self.k_proj = nn.Linear(
                    self.kdim, embed_dim, weight_attr=paddle.ParamAttr(initializer=NumpyArrayInitializer(arr)), bias_attr=bias_attr)
                self.v_proj = nn.Linear(
                    self.vdim, embed_dim, weight_attr=paddle.ParamAttr(initializer=NumpyArrayInitializer(arr)), bias_attr=bias_attr)
            # np.random.seed(5)
            # arr1 = np.random.normal(0, 0.02, size=(384, 768))
            # np.random.seed(5)
            # arr2 = np.random.normal(0, 0.02, size=(384, 768))
            # arr = np.concatenate([arr1, arr2], axis=0)
            # print("single card row seed", arr[:384, :] == arr[384:, :])
            np.random.seed(5)
            arr = np.random.normal(0, 0.02, size=(768, 768))
            self.out_proj = nn.Linear(
                embed_dim, embed_dim, weight_attr=paddle.ParamAttr(initializer=NumpyArrayInitializer(arr)), bias_attr=bias_attr)
            # print(self.q_proj.weight)
            # print(self.k_proj.weight)
            # print(self.v_proj.weight)
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
                np.random.seed(5)
                arr = np.random.normal(0, 0.02, size=(768, 768))
                rank = fleet.worker_index()
                if rank == 0:
                    print("rank is 0")
                    arr_column = arr[:, :384]
                    arr_row = arr[:384, :]
                else:
                    print("rank is 1")
                    arr_column = arr[:, 384:]
                    arr_row = arr[384:, :]
                self.q_proj = paddlenlp.ops.ColumnParallelLiner(
                    (embed_dim, embed_dim),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=paddle.ParamAttr(initializer=NumpyArrayInitializer(arr_column)),
                    bias_attr=bias_attr)
                self.k_proj = paddlenlp.ops.ColumnParallelLiner(
                    (self.kdim, embed_dim),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=paddle.ParamAttr(initializer=NumpyArrayInitializer(arr_column)),
                    bias_attr=bias_attr)
                self.v_proj = paddlenlp.ops.ColumnParallelLiner(
                    (self.vdim, embed_dim),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=paddle.ParamAttr(initializer=NumpyArrayInitializer(arr_column)),
                    bias_attr=bias_attr)

            self.out_proj = paddlenlp.ops.RowParallelLiner(
                (embed_dim, embed_dim),
                topo.mp_info.size,
                input_is_parallel=True,
                param_attr=paddle.ParamAttr(initializer=NumpyArrayInitializer(arr_row)),
                bias_attr=bias_attr)
        # layers.Print(self.k_proj.weight, message="k_proj.weight")
        # layers.Print(self.v_proj.weight, message="v_proj.weight")
        # layers.Print(self.q_proj.weight, message="q_proj.weight")
        # layers.Print(self.out_proj.weight, message="out_proj.weight")

    def _fuse_prepare_qkv(self, query):
        mix_layer = self.qkv_proj(query)
        mix_layer = paddle.reshape_(mix_layer, [0, 0, self.num_heads, 3 * self.head_dim])
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
        # layers.Print(tmp_k, message="tmp_k")
        # layers.Print(tmp_v, message="tmp_v")
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
        # layers.Print(key, message="key")
        # layers.Print(value, message="value")
        # compute q ,k ,v
        if use_cache is False:
            if self.fuse:
                q, k, v = self._fuse_prepare_qkv(query)
            else:
                q, k, v = self._prepare_qkv(query, key, value, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache,
                                               cache)
        # layers.Print(q, message="compute_q")
        # layers.Print(k, message="compute_k")
        # layers.Print(v, message="compute_v")
        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)
        # layers.Print(product, message="product")
        if attn_mask is not None:
            product = product + attn_mask
            #product = product * attn_mask
            #mask_score = (attn_mask - 1.0) * 10000.0
            #product = product + mask_score
        # layers.Print(product, message="product+attn_mask")
        weights = F.softmax(product)
        # layers.Print(weights, message="weights")
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")
            # weights = F.dropout(
            #     weights,
            #     0,
            #     training=self.training,
            #     mode="upscale_in_train")
        # layers.Print(weights, message="dropout")
        # layers.Print(v, message="matmul_v")
        out = tensor.matmul(weights, v)
        # layers.Print(out, message="out1")
        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        # layers.Print(out, message="out1_transpose")
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        # layers.Print(out, message="out1_reshape")
        # project to output
        out = self.out_proj(out)
        # layers.Print(self.out_proj.weight, message="out_proj")
        # layers.Print(out, message="out2")

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
                 topo=None):
        super(TransformerDecoder, self).__init__()

        self.topo = topo
        self.num_layers = num_layers
        self.layers = decoder_layers
        self.norm = norm
        if norm is "LayerNorm":
            self.norm = nn.LayerNorm(hidden_size)
        elif norm is not None:
            raise ValueError("Only support LayerNorm")
        self.checkpoints = []

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
            layers.Print(output, message="output_{}".format(i))
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
                 topo=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        # dropout = 0
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        # attn_dropout = 0
        # act_dropout = 0
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            topo=topo)

        # fuse elementwise_add gelu
        self.fuse_bias_gelu = False
        if topo is None or topo.mp_info.size == 1:
            np.random.seed(5)
            arr1 = np.random.normal(0, 0.02, size=(768, 3072))
            arr2 = np.random.normal(0, 0.02, size=(3072, 768))
            self.linear1 = nn.Linear(
                d_model,
                dim_feedforward,
                paddle.ParamAttr(initializer=NumpyArrayInitializer(arr1)),
                bias_attr=bias_attrs[2])
            self.linear2 = nn.Linear(
                dim_feedforward,
                d_model,
                paddle.ParamAttr(initializer=NumpyArrayInitializer(arr2)),
                bias_attr=bias_attrs[2])
            
        else:
            np.random.seed(5)
            arr1 = np.random.normal(0, 0.02, size=(768, 3072))
            arr2 = np.random.normal(0, 0.02, size=(3072, 768))
            rank = fleet.worker_index()
            if rank == 0:
                print("rank is 0")
                arr1_column = arr1[:, :1536]
                arr2_row = arr2[:1536, :]
            else:
                print("rank is 1")
                arr1_column = arr1[:, 1536:]
                arr2_row = arr2[1536:, :]
            #self.fuse_bias_gelu = True
            self.linear1 = paddlenlp.ops.ColumnParallelLiner(
                (d_model, dim_feedforward),
                topo.mp_info.size,
                gather_out=False,
                param_attr=paddle.ParamAttr(initializer=NumpyArrayInitializer(arr1_column)),
                bias_attr=bias_attrs[2],
                skip_bias_add=self.fuse_bias_gelu)
            self.linear2 = paddlenlp.ops.RowParallelLiner(
                (dim_feedforward, d_model),
                topo.mp_info.size,
                input_is_parallel=True,
                param_attr=paddle.ParamAttr(initializer=NumpyArrayInitializer(arr2_row)),
                bias_attr=bias_attrs[2])
        # layers.Print(self.linear1.weight, message="linear1")
        # layers.Print(self.linear2.weight, message="linear2")
        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, use_cache=False, cache=None):
        residual = tgt
        # layers.Print(residual, message="residual")
        
        if self.normalize_before:
            tgt = self.norm1(tgt)

        # layers.Print(tgt, message="norm1_before")
        if use_cache is False:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    use_cache, cache)
        # layers.Print(tgt, message="attn")
        tgt = residual + self.dropout1(tgt)
        # layers.Print(tgt, message="residual+dropout1")
        if not self.normalize_before:
            tgt = self.norm1(tgt)
        # layers.Print(tgt, message="norm1_after")
        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        # layers.Print(tgt, message="norm2_before")
        if self.fuse_bias_gelu:
            tgt, bias = self.linear1(tgt)
            tgt = paddle.fluid.contrib.layers.fused_elemwise_activation(
                tgt, bias, ['gelu', 'elementwise_add'], save_intermediate_out=False)
        else:
            tgt = self.linear1(tgt)
            tgt = F.gelu(tgt, approximate=True)
        # layers.Print(tgt, message="linear1")
        tgt = self.dropout2(self.linear2(tgt))
        # layers.Print(tgt, message="dropout2")
        tgt = residual + tgt
        # layers.Print(tgt, message="residual + tgt")

        if not self.normalize_before:
            tgt = self.norm2(tgt)
        # layers.Print(tgt, message="norm2 after")
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
                 topo=None):
        super(GPTEmbeddings, self).__init__()
        #if True:
        if topo is None or topo.mp_info.size == 1:
            self.word_embeddings = nn.Embedding(
                vocab_size,
                hidden_size,
                weight_attr=paddle.ParamAttr(
                    name="word_embeddings",
                    initializer=nn.initializer.Normal(
                        mean=0.0, std=initializer_range)))
        else:
            self.word_embeddings = paddlenlp.ops.ParallelEmbedding(
                vocab_size,
                hidden_size,
                topo,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range)))
        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                name="pos_embeddings",
                initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range)))
        # hidden_dropout_prob = 0
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


class GPTPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained GPT models. It provides GPT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "gpt-cpm-large-cn": { # 2.6B
            "vocab_size": 30000,
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 10240,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "gpt-cpm-small-cn-distill": { # 109M
            "vocab_size": 30000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "gpt3-13B-en": { # 13B
            "vocab_size": 50304,
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 128,
            "intermediate_size": 20480,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
        },
        "gpt3-1.3B-en": { # 1.3B
            "vocab_size": 50304,
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 8192,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
        },
        "gpt2-medium-en": { #345M
            "vocab_size": 50304,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
        },
        "gpt2-en": { #117M
            "vocab_size": 50304,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
        },

    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "gpt-cpm-large-cn":
            "https://paddlenlp.bj.bcebos.com/models/transformers/gpt/gpt-cpm-large-cn.pdparams",
            "gpt-cpm-small-cn-distill":
            "https://paddlenlp.bj.bcebos.com/models/transformers/gpt/gpt-cpm-small-cn-distill.pdparams",
            "gpt2-medium-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/gpt/gpt2-medium-en.pdparams",
        }
    }
    base_model_prefix = "gpt"

    def init_weights(self, layer):
        """ Initialization hook """
        # no hook
        return
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.gpt.config["initializer_range"],
                        shape=layer.weight.shape))


@register_base_model
class GPTModel(GPTPretrainedModel):
    """
    The base model of gpt.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 topo=None,
                 split_layers=None):
        super(GPTModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        print("initializer_range", self.initializer_range)
        self.topo = topo
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # hidden_dropout_prob = 0
        self.embeddings = GPTEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, self.initializer_range,
            topo)

        self.pipline_mode = topo is not None and topo.pp_info.size > 1
        if self.pipline_mode:
            pp_size = self.topo.pp_info.size

            # average split
            if split_layers is None:
                assert num_hidden_layers % pp_size == 0
                layer_per_stage = num_hidden_layers // pp_size
                split_layers = [layer_per_stage for _ in range(pp_size)]
            else:
                split_layers = [int(s) for s in split_layers.split(',')]

            assert len(split_layers) == pp_size
            for i in range(pp_size - 1):
                split_layers[i + 1] += split_layers[i]

        decoder_layers = nn.LayerList()
        for i in range(num_hidden_layers):
            DecoderLayer = TransformerDecoderLayer
            if self.pipline_mode:
                stage = pp_size - 1
                for j in range(pp_size):
                    if i < split_layers[j]:
                        stage = j
                        break

                DecoderLayer = paddlenlp.ops.guard(
                    f'gpu:{stage}')(TransformerDecoderLayer)
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
                    topo=topo))

        if self.pipline_mode:
            Decoder = paddlenlp.ops.guard(f'gpu:{self.topo.pp_info.size-1}')(
                TransformerDecoder)
        else:
            Decoder = TransformerDecoder

        self.decoder = Decoder(
            decoder_layers,
            num_hidden_layers,
            norm="LayerNorm",
            hidden_size=hidden_size,
            topo=topo)

        self.apply(self.init_weights)
        self.checkpoints = []

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                cache=None):
        self.checkpoints = []
        if attention_mask is None:
            length = paddle.shape(input_ids)[1]
            # Use bool mask
            attention_mask = paddle.tensor.tril(
                paddle.ones(
                    (length, length),
                    dtype=self.embeddings.word_embeddings.weight.dtype))
        # layers.Print(input_ids, message="input_ids")
        # layers.Print(attention_mask, message="attention_mask")
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
        # layers.Print(position_ids, message="position_ids")
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)
        # layers.Print(embedding_output, message="embedding_output")

        encoder_outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=attention_mask,
            use_cache=use_cache,
            cache=cache)
        self.checkpoints.extend(self.decoder.checkpoints)
        return encoder_outputs


class GPTForPretraining(GPTPretrainedModel):
    """
    The pretraining model of GPT.

    It returns some logits and cached_kvs.
    """

    def __init__(self, gpt):
        super(GPTForPretraining, self).__init__()
        self.gpt = gpt
        self.apply(self.init_weights)

        self.share_param = False
        self.weight = self.gpt.embeddings.word_embeddings.weight
        if not self.share_param:
            self.weight = self.create_parameter(shape=self.weight.shape)

    def parallel_matmul(self, lm_output, logit_weights, parallel_output, topo):
        world_size = topo.mp_info.size
        rank = topo.mp_info.rank

        if world_size > 1:
            model_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
            input_parallel = paddle.distributed.collective._c_identity(
                lm_output, group=model_group)

            logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

            if parallel_output:
                return logits

            return paddle.distributed.collective._c_concat(
                logits, group=model_group)
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
                encoder_outputs,
                self.weight,
                True,
                self.gpt.topo)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPTPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for GPT.

    It calculates the final loss.
    """

    def __init__(self, topo=None):
        super(GPTPretrainingCriterion, self).__init__()
        if topo is None or topo.mp_info.size == 1:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss_func = paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy()

    def forward(self, prediction_scores, masked_lm_labels, loss_mask):
        masked_lm_loss = self.loss_func(
            prediction_scores, masked_lm_labels.unsqueeze(2))

        loss_mask = loss_mask.reshape([-1])
        masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
        loss = masked_lm_loss / loss_mask.sum()
        return loss


class GPTForGreedyGeneration(GPTPretrainedModel):
    """
    The generate model for GPT-2.
    It use the greedy stategy and generate the next word with highest probablity.
    """

    def __init__(self, gpt, max_predict_len):
        super(GPTForGreedyGeneration, self).__init__()
        self.gpt = gpt
        self.max_predict_len = max_predict_len
        self.apply(self.init_weights)

    def model(self,
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
        logits = paddle.matmul(
            encoder_outputs,
            self.gpt.embeddings.word_embeddings.weight,
            transpose_y=True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits

    def forward(self, input_ids, end_id):
        output, cached_kvs = self.model(input_ids, use_cache=True, cache=None)
        src_ids = input_ids
        nid = paddle.argmax(output[:, -1, :], axis=-1).reshape([-1, 1])
        src_ids = paddle.concat([src_ids, nid], axis=1)
        cur_len = 0
        while (cur_len < self.max_predict_len):
            output, cached_kvs = self.model(
                nid, use_cache=True, cache=cached_kvs)

            nid = paddle.argmax(output[:, -1, :], axis=-1).reshape([-1, 1])
            src_ids = paddle.concat([src_ids, nid], axis=1)
            cur_len += 1
            if paddle.max(nid) == end_id:
                break

        return src_ids
