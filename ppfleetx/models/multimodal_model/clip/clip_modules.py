# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import paddle
import paddle.nn as nn
from paddle import tensor
from paddle.fluid import layers
import paddle.nn.functional as F
from paddle.nn import Layer, Linear
from paddle.fluid.data_feeder import convert_dtype
from paddle.nn.layer.transformer import _convert_attention_mask
from paddle.nn.initializer import TruncatedNormal, Constant, Normal


__all__ = ["VisionTransformer"]


trunc_normal_ = TruncatedNormal(std=0.02)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


@paddle.no_grad()
def constant_(x, value):
    temp_value = paddle.full(x.shape, value, x.dtype)
    x.set_value(temp_value)
    return x

@paddle.no_grad()
def normal_(x, mean=0., std=1.):
    temp_value = paddle.normal(mean, std, shape=x.shape)
    x.set_value(temp_value)
    return x

def normal_init(layer, mean=0, std=1, bias=0):
    if hasattr(layer, 'weight') and layer.weight is not None:
        normal_(layer.weight, mean, std)
    else:
        normal_(layer, mean, std)
    if hasattr(layer, 'bias') and layer.bias is not None:
        constant_(layer.bias, bias)


class QuickGELU(Layer):
    """ GELU """
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)

def to_2tuple(x):
    return tuple([x] * 2)

def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_mask=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_mask = attn_mask
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # B= paddle.shape(x)[0]
        N, C = x.shape[1:]
        qkv = (self.qkv(x).reshape(
            (-1, N, 3, self.num_heads, C // self.num_heads)).transpose(
                (2, 0, 3, 1, 4)))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        if self.attn_mask is not None:
            attn_mask = _convert_attention_mask(self.attn_mask, attn.dtype)
            attn = attn + attn_mask
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_mask=None,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=QuickGELU,
        norm_layer="nn.LayerNorm",
        epsilon=1e-5,
    ):
        super().__init__()
        self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        self.attn_mask = attn_mask
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_mask=attn_mask,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def attention(self, x):
        return self.attn(x, attn_mask=self.attn_mask)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Layer):
    def __init__(self,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_mask=None,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer="nn.LayerNorm",
                 epsilon=1e-5,
                 **args):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_mask=attn_mask,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon,
            ) for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding"""
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 patch_bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])

        self.patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size[0],
            stride=patch_size[0],
            bias_attr=patch_bias,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), "Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


class VisionTransformer(nn.Layer):
    """Vision Transformer with support for patch input"""
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_dim=0,
                 width=768,
                 out_dim=512,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer="nn.LayerNorm",
                 pre_norm=False,
                 proj=False,
                 output_cls_token=True,
                 patch_bias=True,
                 epsilon=1e-5,
                 **args):
        super().__init__()
        self.class_dim = class_dim

        self.num_features = self.width = width

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=width,
                                      patch_bias=patch_bias)
        num_patches = self.patch_embed.num_patches

        scale = width**-0.5
        self.class_embedding = self.create_parameter(
            shape=(1, 1, width), default_initializer=Normal(std=scale))
        self.positional_embedding = self.create_parameter(
            shape=(1, num_patches + 1, width),
            default_initializer=Normal(std=scale))
        self.proj = (self.create_parameter(shape=(width, out_dim),
                                           default_initializer=Normal(
                                               std=scale)) if proj else None)
        self.add_parameter("positional_embedding", self.positional_embedding)
        self.add_parameter("class_embedding", self.class_embedding)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.output_cls_token = output_cls_token

        dpr = np.linspace(0, drop_path_rate, depth)

        self.norm_pre = (eval(norm_layer)(width, epsilon=epsilon)
                         if pre_norm else Identity())

        self.blocks = nn.LayerList([
            Block(
                dim=width,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon,
            ) for i in range(depth)
        ])

        self.norm_post = eval(norm_layer)(width, epsilon=epsilon)

        trunc_normal_(self.positional_embedding)
        trunc_normal_(self.class_embedding)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)
        patch_resolution = self.patch_embed.patches_resolution
        class_embedding = self.class_embedding.expand((B, -1, -1))
        x = paddle.concat((class_embedding, x), axis=1)
        x = x + self.positional_embedding
        x = self.pos_drop(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x)
        if self.proj is not None:
            x = self.norm_post(x[:, 0, :])
            x = paddle.matmul(x, self.proj)
            return x
        outs = []
        x = self.norm_post(x)
        B, _, C = x.shape
        patch_token = x[:, 1:].reshape((B, *patch_resolution, C))
        patch_token = patch_token.transpose((0, 3, 1, 2))
        cls_token = x[:, 0]
        if self.output_cls_token:
            out = [patch_token, cls_token]
        else:
            out = patch_token
        outs.append(out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class AttentionPool2D(Layer):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim,
                 dropout=0,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None):
        super(AttentionPool2D, self).__init__()
        self.positional_embedding = paddle.randn((spacial_dim ** 2 + 1, embed_dim)) / embed_dim ** 0.5
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.in_features = embed_dim
        self.q_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = Linear(
            embed_dim, output_dim or embed_dim, weight_attr, bias_attr=bias_attr)

    def _prepare_qkv(self, query, key, value):
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k, v = self.compute_kv(key, value)
        return (q, k, v)
    
    def compute_kv(self, key, value):
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v
    

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3])).transpose((2, 0, 1))  # NCHW -> (HW)NC
        x = paddle.concat([x.mean(axis=0, keepdim=True), x], axis=0)  # (HW+1)NC
        x = x + self.positional_embedding.unsqueeze(axis=1).astype(x.dtype)  # (HW+1)NC
        x = self.multi_head_attention_forward(
            query=x, key=x, value=x,
        )
        return x[0]



    def multi_head_attention_forward(self,
        query, key=None, value=None,
        attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = self._prepare_qkv(query, key, value)
    
        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
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
    
        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)

def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask
