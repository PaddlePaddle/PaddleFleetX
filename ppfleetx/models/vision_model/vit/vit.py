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

from collections.abc import Callable

import os
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..layers.droppath import DropPath
from ..layers.identity import Identity
from ..layers.attention import ViTAttention
from ..layers.embedding import ViTPatchEmbed
from ..layers.mlp import ViTMLP
from ..layers.initializer import xavier_uniform_, zeros_, minus_tens_, pos_normal_, ones_

__all__ = [
    'ViT_tiny_patch16_224',
    'ViT_base_patch16_224',
    'ViT_base_patch16_384',
    'ViT_base_patch32_224',
    'ViT_base_patch32_384',
    'ViT_large_patch16_224',
    'ViT_large_patch16_384',
    'ViT_large_patch32_224',
    'ViT_large_patch32_384',
    'ViT_huge_patch14_224',
    'ViT_huge_patch14_384',
    'ViT_g_patch14_224',
    'ViT_G_patch14_224',
    'ViT_6B_patch14_224',
    'ViT',
]


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm1 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        self.attn = ViTAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm2 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ViTMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Layer):
    """ Vision Transformer with support for patch input
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 representation_size=None,
                 **kwargs):
        super().__init__()
        self.class_num = class_num
        self.representation_size = representation_size

        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = ViTPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_)
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon) for i in range(depth)
        ])

        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        # Classifier head
        if self.representation_size is not None:
            self.head0 = nn.Linear(embed_dim, representation_size)
            self.tanh = nn.Tanh()
            self.head = nn.Linear(representation_size,
                                  class_num) if class_num > 0 else Identity()
            xavier_uniform_(self.head0.weight)
            zeros_(self.head0.bias)
            xavier_uniform_(self.head.weight)
            minus_tens_(self.head.bias)
        else:
            self.head = nn.Linear(embed_dim,
                                  class_num) if class_num > 0 else Identity()
            zeros_(self.head.weight)
            zeros_(self.head.bias)

        pos_normal_(self.pos_embed)
        zeros_(self.cls_token)
        self.apply(self._init_weights)

        pretrained_configs = kwargs.pop('pretrained', None)
        if pretrained_configs is not None:
            self.load_pretrained(**pretrained_configs)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        # B = x.shape[0]
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        if self.representation_size is not None:
            x = self.tanh(self.head0(x))
        x = self.head(x)
        return x

    def load_pretrained(self, prefix_path, finetune=False):
        if not os.path.exists(prefix_path + '.pdparams'):
            raise ValueError("Model pretrain path {} does not "
                             "exists.".format(prefix_path))

        state_dict = self.state_dict()
        param_state_dict = paddle.load(prefix_path + ".pdparams")

        # for FP16 saving pretrained weight
        for key, value in param_state_dict.items():
            param_state_dict[key] = param_state_dict[key].astype(
                paddle.float32)

        if not finetune:
            self.set_dict(param_state_dict)
            return

        for k in ['head0.weight', 'head0.bias', 'head.weight', 'head.bias']:
            if k in param_state_dict:
                print(f"Removing key {k} from pretrained checkpoint")
                del param_state_dict[k]

        # interpolate position embedding
        pos_embed_checkpoint = param_state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = self.patch_embed.num_patches
        num_extra_tokens = self.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**
                        0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = paddle.transpose(
            pos_tokens.reshape([-1, orig_size, orig_size, embedding_size]),
            perm=[0, 3, 1, 2])
        dtype = pos_tokens.dtype
        pos_tokens = paddle.nn.functional.interpolate(
            pos_tokens.astype(paddle.float32),
            size=(new_size, new_size),
            mode='bicubic',
            align_corners=False).astype(dtype)
        pos_tokens = paddle.transpose(
            pos_tokens, perm=[0, 2, 3, 1]).flatten(1, 2)
        new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
        param_state_dict['pos_embed'] = new_pos_embed

        self.set_dict(param_state_dict)
        return

def ViT_tiny_patch16_224(**kwargs):
    model = ViT(patch_size=16,
                embed_dim=192,
                depth=12,
                num_heads=3,
                mlp_ratio=4,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=192,
                **kwargs)
    return model

def ViT_base_patch16_224(**kwargs):
    model = ViT(patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=768,
                **kwargs)
    return model


def ViT_base_patch16_384(**kwargs):
    model = ViT(img_size=384,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=None,
                **kwargs)
    return model


def ViT_base_patch32_224(**kwargs):
    model = ViT(patch_size=32,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=768,
                **kwargs)
    return model


def ViT_base_patch32_384(**kwargs):
    model = ViT(img_size=384,
                patch_size=32,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=None,
                **kwargs)
    return model


def ViT_large_patch16_224(**kwargs):
    model = ViT(patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=1024,
                **kwargs)
    return model


def ViT_large_patch16_384(**kwargs):
    model = ViT(img_size=384,
                patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=None,
                **kwargs)
    return model


def ViT_large_patch32_224(**kwargs):
    model = ViT(patch_size=32,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=1024,
                **kwargs)
    return model


def ViT_large_patch32_384(**kwargs):
    model = ViT(img_size=384,
                patch_size=32,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=None,
                **kwargs)
    return model


def ViT_huge_patch14_224(**kwargs):
    model = ViT(patch_size=14,
                embed_dim=1280,
                depth=32,
                num_heads=16,
                mlp_ratio=4,
                representation_size=1280,
                **kwargs)
    return model


def ViT_huge_patch14_384(**kwargs):
    model = ViT(img_size=384,
                patch_size=14,
                embed_dim=1280,
                depth=32,
                num_heads=16,
                mlp_ratio=4,
                representation_size=None,
                **kwargs)
    return model


def ViT_g_patch14_224(**kwargs):
    model = ViT(img_size=224,
                patch_size=14,
                embed_dim=1408,
                depth=40,
                num_heads=16,
                mlp_ratio=4.364,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=1408,
                **kwargs)
    return model


def ViT_G_patch14_224(**kwargs):
    model = ViT(img_size=224,
                patch_size=14,
                embed_dim=1664,
                depth=48,
                num_heads=16,
                mlp_ratio=4.9231,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=1664,
                **kwargs)
    return model


def ViT_6B_patch14_224(**kwargs):
    model = ViT(img_size=224,
                patch_size=14,
                embed_dim=2320,
                depth=80,
                num_heads=16,
                mlp_ratio=4.955,
                qkv_bias=True,
                epsilon=1e-6,
                representation_size=2320,
                **kwargs)
    return model
