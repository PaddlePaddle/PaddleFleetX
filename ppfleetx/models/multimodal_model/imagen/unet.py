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

import math
from pathlib import Path
from functools import partial

import paddle
from paddle import nn
from paddle import nn, einsum
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute

from .utils import (zeros_, zero_init_, default, exists, cast_tuple, l2norm,
                    resize_image_to, prob_mask_like, masked_mean, Identity,
                    repeat, repeat_many, Rearrange, rearrange, rearrange_many,
                    EinopsToAndFrom, Parallel, Always, print_once)

from ppfleetx.models.language_model.t5.modeling import finfo


class LayerNorm(nn.Layer):
    def __init__(self, feats, stable=False, dim=-1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = self.create_parameter(
            [feats, *((1, ) * (-dim - 1))],
            default_initializer=nn.initializer.Constant(value=1.))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(axis=dim, keepdim=True).detach()

        eps = 1e-5 if x.dtype == paddle.float32 else 1e-3
        var = paddle.var(x, axis=dim, unbiased=False, keepdim=True)
        mean = paddle.mean(x, axis=dim, keepdim=True)

        return (x - mean) * (
            var + eps).rsqrt().cast(dtype) * self.g.cast(dtype)


ChanLayerNorm = partial(LayerNorm, dim=-3)


class Residual(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# attention pooling


class PerceiverAttention(nn.Layer):
    def __init__(self, *, dim, dim_head=64, heads=8, cosine_sim_attn=False):
        super().__init__()
        self.scale = dim_head**-0.5 if not cosine_sim_attn else 1
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias_attr=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias_attr=False)

        self.to_out = nn.Sequential(
            nn.Linear(
                inner_dim, dim, bias_attr=False), nn.LayerNorm(dim))

    def forward(self, x, latents, mask=None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = paddle.concat((x, latents), axis=-2)
        k, v = self.to_kv(kv_input).chunk(2, axis=-1)

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # similarities and masking

        sim = einsum('... i d, ... j d  -> ... i j', q,
                     k) * self.cosine_sim_scale

        if exists(mask):
            max_neg_value = -finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = paddle.where(mask == 0, paddle.to_tensor(max_neg_value), sim)

        # attention

        attn = F.softmax(sim, axis=-1, dtype=paddle.float32)
        attn = attn.cast(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        B, H, N, D = out.shape
        out = out.transpose([0, 2, 1, 3]).reshape([B, N, -1])
        return self.to_out(out)


class PerceiverResampler(nn.Layer):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            num_latents=64,
            num_latents_mean_pooled=4,  # number of latents derived from mean pooled representation of the sequence
            max_seq_len=512,
            ff_mult=4,
            cosine_sim_attn=False):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = self.create_parameter(
            [num_latents, dim], default_initializer=nn.initializer.Normal())

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange(
                    'b (n d) -> b n d', n=num_latents_mean_pooled))

        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(
                nn.LayerList([
                    PerceiverAttention(
                        dim=dim,
                        dim_head=dim_head,
                        heads=heads,
                        cosine_sim_attn=cosine_sim_attn), FeedForward(
                            dim=dim, mult=ff_mult)
                ]))

    def forward(self, x, mask=None):
        n = x.shape[1]
        pos_emb = self.pos_emb(paddle.arange(n))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(
                x, axis=1, mask=paddle.ones(
                    x.shape[:2], dtype=paddle.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(
                meanpooled_seq)
            latents = paddle.concat((meanpooled_latents, latents), axis=-2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        return latents


# attention


class Attention(nn.Layer):
    def __init__(
            self,
            dim,
            *,
            dim_head=64,
            heads=8,
            context_dim=None,
            cosine_sim_attn=False,
            use_recompute=False, ):
        super().__init__()
        self.use_recompute = use_recompute
        self.scale = dim_head**-0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = self.create_parameter(
            [2, dim_head], default_initializer=nn.initializer.Normal())
        self.to_q = nn.Linear(dim, inner_dim, bias_attr=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias_attr=False)

        self.to_context = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(
                context_dim, dim_head * 2)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(
                inner_dim, dim, bias_attr=False), LayerNorm(dim))

    def forward(self, x, context=None, mask=None, attn_bias=None):
        if self.use_recompute:
            return recompute(self._forward, x, context, mask, attn_bias)
        else:
            return self._forward(x, context, mask, attn_bias)

    def _forward(self, x, context=None, mask=None, attn_bias=None):
        b, n = x.shape[:2]

        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, axis=-1))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(axis=-2), 'd -> b 1 d', b=b)
        k = paddle.concat((nk, k), axis=-2)
        v = paddle.concat((nv, v), axis=-2)

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, axis=-1)
            k = paddle.concat((ck, k), axis=-2)
            v = paddle.concat((cv, v), axis=-2)

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.cosine_sim_scale

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = paddle.where(mask == 0, paddle.to_tensor(max_neg_value), sim)

        # attention

        attn = F.softmax(sim, axis=-1, dtype=paddle.float32)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# decoder


def Upsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(
            scale_factor=2, mode='nearest'),
        nn.Conv2D(
            dim, dim_out, 3, padding=1))


class PixelShuffleUpsample(nn.Layer):
    """
    code shared by @MalumaDev at DALLE2 for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2D(dim, dim_out * 4, 1)

        self.net = nn.Sequential(conv, nn.Silu(), nn.PixelShuffle(2))

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = paddle.empty([o // 4, i, h, w])
        nn.initializer.KaimingUniform(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.set_value(conv_weight)
        zeros_(conv.bias)

    def forward(self, x):
        return self.net(x)


def Downsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange(
            'b c (h s1) (w s2) -> b (c s1 s2) h w', s1=2, s2=2),
        nn.Conv2D(dim * 4, dim_out, 1))


class SinusoidalPosEmb(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(paddle.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        return paddle.concat((emb.sin(), emb.cos()), axis=-1)


class LearnedSinusoidalPosEmb(nn.Layer):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = self.create_parameter(
            [half_dim], default_initializer=nn.initializer.Normal())

    def forward(self, x):
        x = x[:, None]
        freqs = x * self.weights[None, :] * 2 * math.pi
        fouriered = paddle.concat((freqs.sin(), freqs.cos()), axis=-1)
        fouriered = paddle.concat((x, fouriered), axis=-1)
        return fouriered


class Block(nn.Layer):
    def __init__(self, dim, dim_out, groups=8, norm=True):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.Silu()
        self.project = nn.Conv2D(dim, dim_out, 3, padding=1)

    def forward(self, x, scale_shift=None):
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)


class ResnetBlock(nn.Layer):
    def __init__(self,
                 dim,
                 dim_out,
                 *,
                 cond_dim=None,
                 time_cond_dim=None,
                 groups=8,
                 linear_attn=False,
                 use_gca=False,
                 squeeze_excite=False,
                 use_recompute=False,
                 **attn_kwargs):
        super().__init__()

        self.time_mlp = None
        self.use_recompute = use_recompute

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.Silu(), nn.Linear(time_cond_dim, dim_out * 2))

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = attn_klass(
                dim=dim_out, context_dim=cond_dim, **attn_kwargs)

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.gca = GlobalContext(
            dim_in=dim_out, dim_out=dim_out) if use_gca else Always(1)

        self.res_conv = nn.Conv2D(dim, dim_out,
                                  1) if dim != dim_out else Identity()

    def forward(self, x, time_emb=None, cond=None):
        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = time_emb[:, :, None, None]
            scale_shift = time_emb.chunk(2, axis=1)

        h = self.block1(x)

        if exists(self.cross_attn):
            assert exists(cond)
            h = h.transpose([0, 2, 3, 1])
            n, b, c, *_ = h.shape
            h = h.reshape([n, b * c, -1])
            h = self.cross_attn(h, context=cond) + h
            h = h.reshape([n, b, c, -1])
            h = h.transpose([0, 3, 1, 2])

        h = self.block2(h, scale_shift=scale_shift)

        h = h * self.gca(h)

        return h + self.res_conv(x)


class CrossAttention(nn.Layer):
    def __init__(self,
                 dim,
                 *,
                 context_dim=None,
                 dim_head=64,
                 heads=8,
                 norm_context=False,
                 cosine_sim_attn=False):
        super().__init__()
        self.scale = dim_head**-0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(
            context_dim) if norm_context else Identity()

        self.null_kv = self.create_parameter(
            [2, dim_head], default_initializer=nn.initializer.Normal())
        self.to_q = nn.Linear(dim, inner_dim, bias_attr=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias_attr=False)

        self.to_out = nn.Sequential(
            nn.Linear(
                inner_dim, dim, bias_attr=False), LayerNorm(dim))

    def forward(self, x, context, mask=None):
        b, n = x.shape[:2]

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, axis=-1))

        q, k, v = rearrange_many(
            (q, k, v), 'b n (h d) -> b h n d', h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(
            self.null_kv.unbind(axis=-2), 'd -> b h 1 d', h=self.heads, b=b)

        k = paddle.concat((nk, k), axis=-2)
        v = paddle.concat((nv, v), axis=-2)

        q = q * self.scale

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # similarities

        sim = einsum('b h i d, b h j d -> b h i j', q,
                     k) * self.cosine_sim_scale

        # masking

        max_neg_value = -finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = paddle.where(mask == 0, paddle.to_tensor(max_neg_value), sim)

        attn = F.softmax(sim, axis=-1, dtype=paddle.float32)
        attn = attn.cast(sim.dtype)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearCrossAttention(CrossAttention):
    def forward(self, x, context, mask=None):
        b, n = x.shape[:2]

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, axis=-1))

        q, k, v = rearrange_many(
            (q, k, v), 'b n (h d) -> (b h) n d', h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(
            self.null_kv.unbind(axis=-2), 'd -> (b h) 1 d', h=self.heads, b=b)

        k = paddle.concat((nk, k), axis=-2)
        v = paddle.concat((nv, v), axis=-2)

        # masking

        max_neg_value = -finfo(x.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = paddle.where(mask == 0, paddle.to_tensor(max_neg_value), k)
            v = paddle.where(mask == 0, paddle.to_tensor(0.), v)

        # linear attention

        q = F.softmax(q, axis=-1)
        k = F.softmax(k, axis=-2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(out)


class LinearAttention(nn.Layer):
    def __init__(self,
                 dim,
                 dim_head=32,
                 heads=8,
                 dropout=0.05,
                 context_dim=None,
                 **kwargs):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.Silu()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2D(
                dim, inner_dim, 1, bias_attr=False),
            nn.Conv2D(
                inner_dim,
                inner_dim,
                3,
                bias_attr=False,
                padding=1,
                groups=inner_dim))

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2D(
                dim, inner_dim, 1, bias_attr=False),
            nn.Conv2D(
                inner_dim,
                inner_dim,
                3,
                bias_attr=False,
                padding=1,
                groups=inner_dim))

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2D(
                dim, inner_dim, 1, bias_attr=False),
            nn.Conv2D(
                inner_dim,
                inner_dim,
                3,
                bias_attr=False,
                padding=1,
                groups=inner_dim))

        self.to_context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(
                context_dim, inner_dim * 2,
                bias_attr=False)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv2D(
                inner_dim, dim, 1, bias_attr=False), ChanLayerNorm(dim))

    def forward(self, fmap, context=None):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = rearrange_many(
            (q, k, v), 'b (h c) x y -> (b h) (x y) c', h=h)

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, axis=-1)
            ck, cv = rearrange_many((ck, cv), 'b n (h d) -> (b h) n d', h=h)
            k = paddle.concat((k, ck), axis=-2)
            v = paddle.concat((v, cv), axis=-2)

        q = F.softmax(q, axis=-1)
        k = F.softmax(k, axis=-2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)


class GlobalContext(nn.Layer):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(self, *, dim_in, dim_out):
        super().__init__()
        self.to_k = nn.Conv2D(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2D(dim_in, hidden_dim, 1),
            nn.Silu(), nn.Conv2D(hidden_dim, dim_out, 1), nn.Sigmoid())

    def forward(self, x):
        context = self.to_k(x)
        x, context = rearrange_many((x, context), 'b n ... -> b n (...)')
        out = einsum('b i n, b c n -> b c i', F.softmax(context, axis=-1), x)
        out = out[:, :, :, None]
        return self.net(out)


def FeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(
            dim, hidden_dim, bias_attr=False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(
            hidden_dim, dim, bias_attr=False))


def ChanFeedForward(
        dim, mult=2
):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2D(
            dim, hidden_dim, 1, bias_attr=False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv2D(
            hidden_dim, dim, 1, bias_attr=False))


class TransformerBlock(nn.Layer):
    def __init__(
            self,
            dim,
            *,
            depth=1,
            heads=8,
            dim_head=32,
            ff_mult=2,
            context_dim=None,
            cosine_sim_attn=False,
            use_recompute=False, ):
        super().__init__()
        self.layers = nn.LayerList([])

        for _ in range(depth):
            self.layers.append(
                nn.LayerList([
                    Attention(
                        dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        context_dim=context_dim,
                        cosine_sim_attn=cosine_sim_attn,
                        use_recompute=use_recompute), FeedForward(
                            dim=dim, mult=ff_mult)
                ]))

    def forward(self, x, context=None):
        x = x.transpose([0, 2, 3, 1])
        n, b, c, *_ = x.shape
        x = x.reshape([n, b * c, -1])
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        x = x.reshape([n, b, c, -1])
        x = x.transpose([0, 3, 1, 2])
        return x


class LinearAttentionTransformerBlock(nn.Layer):
    def __init__(self,
                 dim,
                 *,
                 depth=1,
                 heads=8,
                 dim_head=32,
                 ff_mult=2,
                 context_dim=None,
                 **kwargs):
        super().__init__()
        self.layers = nn.LayerList([])

        for _ in range(depth):
            self.layers.append(
                nn.LayerList([
                    LinearAttention(
                        dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        context_dim=context_dim), ChanFeedForward(
                            dim=dim, mult=ff_mult)
                ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x


class CrossEmbedLayer(nn.Layer):
    def __init__(self, dim_in, kernel_sizes, dim_out=None, stride=2):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.LayerList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2D(
                    dim_in,
                    dim_scale,
                    kernel,
                    stride=stride,
                    padding=(kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return paddle.concat(fmaps, axis=1)


class UpsampleCombiner(nn.Layer):
    def __init__(self,
                 dim,
                 *,
                 enabled=False,
                 dim_ins=tuple(),
                 dim_outs=tuple()):
        super().__init__()
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.LayerList([
            Block(dim_in, dim_out)
            for dim_in, dim_out in zip(dim_ins, dim_outs)
        ])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps=None):
        target_size = x.shape[-1]

        fmaps = default(fmaps, tuple())

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return paddle.concat((x, *outs), axis=1)


class Unet(nn.Layer):
    def __init__(self,
                 *,
                 dim,
                 image_embed_dim=1024,
                 text_embed_dim=1024,
                 num_resnet_blocks=1,
                 cond_dim=None,
                 num_image_tokens=4,
                 num_time_tokens=2,
                 learned_sinu_pos_emb_dim=16,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 cond_images_channels=0,
                 channels=3,
                 channels_out=None,
                 attn_dim_head=64,
                 attn_heads=8,
                 ff_mult=2.,
                 lowres_cond=False,
                 layer_attns=True,
                 layer_attns_depth=1,
                 layer_mid_attns_depth=1,
                 layer_attns_add_text_cond=True,
                 attend_at_middle=True,
                 layer_cross_attns=True,
                 use_linear_attn=False,
                 use_linear_cross_attn=False,
                 cond_on_text=True,
                 max_text_len=256,
                 init_dim=None,
                 resnet_groups=8,
                 init_conv_kernel_size=7,
                 init_cross_embed=True,
                 init_cross_embed_kernel_sizes=(3, 7, 15),
                 cross_embed_downsample=False,
                 cross_embed_downsample_kernel_sizes=(2, 4),
                 attn_pool_text=True,
                 attn_pool_num_latents=32,
                 dropout=0.,
                 memory_efficient=False,
                 init_conv_to_final_conv_residual=False,
                 use_global_context_attn=True,
                 scale_skip_connection=True,
                 final_resnet_block=True,
                 final_conv_kernel_size=3,
                 cosine_sim_attn=False,
                 self_cond=False,
                 combine_upsample_fmaps=False,
                 pixel_shuffle_upsample=True,
                 use_recompute=False):
        super().__init__()

        self.use_recompute = use_recompute
        # guide researchers

        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'

        if dim < 128:
            print_once(
                'The base dimension of your u-net should ideally be no smaller than 128, as recommended by a professional DDPM trainer https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/'
            )

        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        init_channels = channels * (1 + int(lowres_cond) + int(self_cond))
        init_dim = default(init_dim, dim)

        self.self_cond = self_cond

        # optional image conditioning

        self.has_cond_image = cond_images_channels > 0
        self.cond_images_channels = cond_images_channels

        init_channels += cond_images_channels

        # initial convolution

        self.init_conv = CrossEmbedLayer(
            init_channels,
            dim_out=init_dim,
            kernel_sizes=init_cross_embed_kernel_sizes,
            stride=1) if init_cross_embed else nn.Conv2D(
                init_channels,
                init_dim,
                init_conv_kernel_size,
                padding=init_conv_kernel_size // 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1)

        # embedding time for log(snr) noise from continuous version

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1

        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim), nn.Silu())

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim))

        # project to time tokens as well as time hiddens

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange(
                'b (n d) -> b n d', n=num_time_tokens))

        # low res aug noise conditioning

        self.lowres_cond = lowres_cond

        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
                nn.Linear(learned_sinu_pos_emb_dim + 1, time_cond_dim),
                nn.Silu())

            self.to_lowres_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim))

            self.to_lowres_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange(
                    'b (n d) -> b n d', n=num_time_tokens))

        # normalizations

        self.norm_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text:
            assert exists(
                text_embed_dim
            ), 'text_embed_dim must be given to the unet if cond_on_text is True'
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # finer control over whether to condition on text encodings

        self.cond_on_text = cond_on_text

        # attention pooling

        self.attn_pool = PerceiverResampler(
            dim=cond_dim,
            depth=2,
            dim_head=attn_dim_head,
            heads=attn_heads,
            num_latents=attn_pool_num_latents,
            cosine_sim_attn=cosine_sim_attn) if attn_pool_text else None

        # for classifier free guidance

        self.max_text_len = max_text_len

        self.null_text_embed = self.create_parameter(
            [1, max_text_len, cond_dim],
            default_initializer=nn.initializer.Normal())
        self.null_text_hidden = self.create_parameter(
            [1, time_cond_dim], default_initializer=nn.initializer.Normal())

        # for non-attention based text conditioning at all points in the network where time is also conditioned

        self.to_text_non_attn_cond = None

        if cond_on_text:
            self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, time_cond_dim),
                nn.Silu(), nn.Linear(time_cond_dim, time_cond_dim))

        # attention related params

        attn_kwargs = dict(
            heads=attn_heads,
            dim_head=attn_dim_head,
            cosine_sim_attn=cosine_sim_attn,
            use_recompute=use_recompute)

        num_layers = len(in_out)

        # resnet block klass

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)

        resnet_klass = partial(ResnetBlock, **attn_kwargs)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        use_linear_attn = cast_tuple(use_linear_attn, num_layers)
        use_linear_cross_attn = cast_tuple(use_linear_cross_attn, num_layers)

        assert all([
            layers == num_layers
            for layers in list(
                map(len, (resnet_groups, layer_attns, layer_cross_attns)))
        ])

        # downsample klass

        downsample_klass = Downsample

        if cross_embed_downsample:
            downsample_klass = partial(
                CrossEmbedLayer,
                kernel_sizes=cross_embed_downsample_kernel_sizes)

        # initial resnet block (for memory efficient unet)

        self.init_resnet_block = resnet_klass(
            init_dim,
            init_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[0],
            use_gca=use_global_context_attn) if memory_efficient else None

        # scale for resnet skip connections

        self.skip_connect_scale = 1. if not scale_skip_connection else (2
                                                                        **-0.5)

        # layers

        self.downs = nn.LayerList([])
        self.ups = nn.LayerList([])
        num_resolutions = len(in_out)

        layer_params = [
            num_resnet_blocks, resnet_groups, layer_attns, layer_attns_depth,
            layer_cross_attns, use_linear_attn, use_linear_cross_attn
        ]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers

        skip_connect_dims = []  # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups,
                  layer_attn, layer_attn_depth, layer_cross_attn,
                  layer_use_linear_attn, layer_use_linear_cross_attn
                  ) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet

            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet

            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(
                    current_dim, dim_out) if not is_last else Parallel(
                        nn.Conv2D(
                            dim_in, dim_out, 3, padding=1),
                        nn.Conv2D(dim_in, dim_out, 1))

            self.downs.append(
                nn.LayerList([
                    pre_downsample, resnet_klass(
                        current_dim,
                        current_dim,
                        cond_dim=layer_cond_dim,
                        linear_attn=layer_use_linear_cross_attn,
                        time_cond_dim=time_cond_dim,
                        groups=groups,
                        use_recompute=use_recompute), nn.LayerList([
                            ResnetBlock(
                                current_dim,
                                current_dim,
                                time_cond_dim=time_cond_dim,
                                groups=groups,
                                use_gca=use_global_context_attn,
                                use_recompute=use_recompute)
                            for _ in range(layer_num_resnet_blocks)
                        ]), transformer_block_klass(
                            dim=current_dim,
                            depth=layer_attn_depth,
                            ff_mult=ff_mult,
                            context_dim=cond_dim,
                            **attn_kwargs), post_downsample
                ]))

        # middle layers

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
            use_recompute=use_recompute)
        self.mid_attn = TransformerBlock(
            mid_dim, depth=layer_mid_attns_depth,
            **attn_kwargs) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
            use_recompute=use_recompute)

        # upsample klass

        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        # upsampling layers

        upsample_fmap_dims = []

        for ind, (
            (dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn,
                layer_attn_depth, layer_cross_attn, layer_use_linear_attn,
                layer_use_linear_cross_attn
        ) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            skip_connect_dim = skip_connect_dims.pop()

            upsample_fmap_dims.append(dim_out)

            self.ups.append(
                nn.LayerList([
                    resnet_klass(
                        dim_out + skip_connect_dim,
                        dim_out,
                        cond_dim=layer_cond_dim,
                        linear_attn=layer_use_linear_cross_attn,
                        time_cond_dim=time_cond_dim,
                        groups=groups,
                        use_recompute=use_recompute), nn.LayerList([
                            ResnetBlock(
                                dim_out + skip_connect_dim,
                                dim_out,
                                time_cond_dim=time_cond_dim,
                                groups=groups,
                                use_gca=use_global_context_attn,
                                use_recompute=use_recompute)
                            for _ in range(layer_num_resnet_blocks)
                        ]), transformer_block_klass(
                            dim=dim_out,
                            depth=layer_attn_depth,
                            ff_mult=ff_mult,
                            context_dim=cond_dim,
                            **attn_kwargs), upsample_klass(dim_out, dim_in)
                    if not is_last or memory_efficient else Identity()
                ]))

        # whether to combine feature maps from all upsample blocks before final resnet block out

        self.upsample_combiner = UpsampleCombiner(
            dim=dim,
            enabled=combine_upsample_fmaps,
            dim_ins=upsample_fmap_dims,
            dim_outs=dim)

        # whether to do a final residual from initial conv to the final resnet block out

        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (
            dim if init_conv_to_final_conv_residual else 0)

        # final optional resnet block and convolution out

        self.final_res_block = ResnetBlock(
            final_conv_dim,
            dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[0],
            use_gca=True,
            use_recompute=use_recompute) if final_resnet_block else None

        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        final_conv_dim_in += (channels if lowres_cond else 0)

        self.final_conv = nn.Conv2D(
            final_conv_dim_in,
            self.channels_out,
            final_conv_kernel_size,
            padding=final_conv_kernel_size // 2)

        zero_init_(self.final_conv)

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(self, *, text_embed_dim, channels, channels_out,
                              cond_on_text):
        if channels == self.channels and \
            cond_on_text == self.cond_on_text and \
            text_embed_dim == self._locals['text_embed_dim'] and \
            channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            text_embed_dim=text_embed_dim,
            channels=channels,
            channels_out=channels_out,
            cond_on_text=cond_on_text)

        return self.__class__(**{ ** self._locals, ** updated_kwargs})

    # methods for returning the full unet config as well as its parameter state

    def to_config_and_state_dict(self):
        return self._locals, self.state_dict()

    # class method for rehydrating the unet from its config and state dict

    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        unet = klass(**config)
        unet.load_state_dict(state_dict)
        return unet

    # methods for persisting unet to disk

    def persist_to_file(self, path):
        path = Path(path)
        path.parents[0].mkdir(exist_ok=True, parents=True)

        config, state_dict = self.to_config_and_state_dict()
        pkg = dict(config=config, state_dict=state_dict)
        paddle.save(pkg, str(path))

    # class method for rehydrating the unet from file saved with `persist_to_file`

    @classmethod
    def hydrate_from_file(klass, path):
        path = Path(path)
        assert path.exists()
        pkg = paddle.load(str(path))

        assert 'config' in pkg and 'state_dict' in pkg
        config, state_dict = pkg['config'], pkg['state_dict']

        return Unet.from_config_and_state_dict(config, state_dict)

    # forward with classifier free guidance

    def forward_with_cond_scale(self, *args, cond_scale=1., **kwargs):
        #print("forward_with_cond_scale.args[1]: ", args[1])
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self,
                x,
                time,
                *,
                lowres_cond_img=None,
                lowres_noise_times=None,
                text_embeds=None,
                text_mask=None,
                self_cond=None,
                cond_images=None,
                cond_drop_prob=0.,
                use_recompute=False):
        batch_size = x.shape[0]

        # condition on self

        if self.self_cond:
            self_cond = default(self_cond, lambda: paddle.zeros_like(x))
            x = paddle.concat((x, self_cond), axis=1)

        # add low resolution conditioning, if present

        assert not (self.lowres_cond and not exists(lowres_cond_img)
                    ), 'low resolution conditioning image must be present'
        assert not (self.lowres_cond and not exists(lowres_noise_times)
                    ), 'low resolution conditioning noise time must be present'

        if exists(lowres_cond_img):
            x = paddle.concat((x, lowres_cond_img), axis=1)

        # condition on input image

        assert not (
            self.has_cond_image ^ exists(cond_images)
        ), 'you either requested to condition on an image on the unet, but the conditioning image is not supplied, or vice versa'

        if exists(cond_images):
            assert cond_images.shape[
                1] == self.cond_images_channels, 'the number of channels on the conditioning image you are passing in does not match what you specified on initialiation of the unet'
            cond_images = resize_image_to(cond_images, x.shape[-1])
            x = paddle.concat((cond_images, x), axis=1)

        # initial convolution

        x = self.init_conv(x)

        # init conv residual

        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # time conditioning

        time_hiddens = self.to_time_hiddens(time)

        # derive time tokens

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)
        if use_recompute:
            t.stop_gradient = True
        # add lowres time conditioning to time hiddens
        # and add lowres time tokens along sequence dimension for attention

        if self.lowres_cond:
            lowres_time_hiddens = self.to_lowres_time_hiddens(
                lowres_noise_times)
            lowres_time_tokens = self.to_lowres_time_tokens(
                lowres_time_hiddens)
            lowres_t = self.to_lowres_time_cond(lowres_time_hiddens)

            t = t + lowres_t
            time_tokens = paddle.concat(
                (time_tokens, lowres_time_tokens), axis=-2)

        # text conditioning

        text_tokens = None

        if exists(text_embeds) and self.cond_on_text:

            # conditional dropout

            text_keep_mask = prob_mask_like((batch_size, ), 1 - cond_drop_prob)

            text_keep_mask_embed = text_keep_mask[:, None, None]
            text_keep_mask_hidden = text_keep_mask[:, None]

            # calculate text embeds

            text_tokens = self.text_to_cond(text_embeds)

            text_tokens = text_tokens[:, :self.max_text_len]

            if exists(text_mask):
                text_mask = text_mask[:, :self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, remainder),
                                    data_format='NLC')

            if exists(text_mask):
                text_mask = text_mask[:, :, None].cast('float32')
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder),
                                      data_format='NLC')

                text_keep_mask_embed = text_mask.cast(
                    bool) & text_keep_mask_embed

            null_text_embed = self.null_text_embed.cast(text_tokens.dtype)

            text_tokens = paddle.where(text_keep_mask_embed, text_tokens,
                                       null_text_embed)

            if exists(self.attn_pool):
                text_tokens = self.attn_pool(text_tokens)

            # extra non-attention conditioning by projecting and then summing text embeddings to time
            # termed as text hiddens

            mean_pooled_text_tokens = text_tokens.mean(axis=-2)

            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)

            null_text_hidden = self.null_text_hidden.cast(t.dtype)

            text_hiddens = paddle.where(text_keep_mask_hidden, text_hiddens,
                                        null_text_hidden)

            t = t + text_hiddens

        # main conditioning tokens (c)

        c = time_tokens if not exists(text_tokens) else paddle.concat(
            (time_tokens, text_tokens), axis=-2)

        # normalize conditioning tokens

        c = self.norm_cond(c)
        if use_recompute:
            c.stop_gradient = True

        # initial resnet block (for memory efficient unet)

        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)

        hiddens = []

        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            x = attn_block(x, c)
            hiddens.append(x)

            if exists(post_downsample):
                x = post_downsample(x)

        x = self.mid_block1(x, t, c)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, t, c)

        add_skip_connection = lambda x: paddle.concat((x, hiddens.pop() * self.skip_connect_scale), axis=1)

        up_hiddens = []

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            x = attn_block(x, c)
            up_hiddens.append(x)
            x = upsample(x)

        x = self.upsample_combiner(x, up_hiddens)

        if self.init_conv_to_final_conv_residual:
            x = paddle.concat((x, init_conv_residual), axis=1)

        if exists(self.final_res_block):
            x = self.final_res_block(x, t)

        if exists(lowres_cond_img):
            x = paddle.concat((x, lowres_cond_img), axis=1)

        return self.final_conv(x)
