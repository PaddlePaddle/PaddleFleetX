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

# Code based on:
# https://github.com/PaddlePaddle/PLSC/blob/master/plsc/models/vision_transformer.py

from collections.abc import Callable

import os
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant, Normal, XavierUniform

__all__ = [
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

mlp_bias_normal_ = Normal(std=1e-6)
pos_normal_ = Normal(std=0.02)
xavier_uniform_ = XavierUniform()
zeros_ = Constant(value=0.)
minus_tens_ = Constant(value=-10.)
ones_ = Constant(value=1.)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    if x.dtype == paddle.float16:
        random_tensor = keep_prob + paddle.rand(
            shape, dtype=paddle.float32).astype(x.dtype)
    else:
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ViTPatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = img_size if isinstance(img_size, tuple) else (img_size,
                                                                 img_size)
        patch_size = patch_size if isinstance(patch_size, tuple) else (
            patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_uniform_(m.weight)
            mlp_bias_normal_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTAttention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_uniform_(m.weight)
            zeros_(m.bias)

    def forward(self, x):
        # B= paddle.shape(x)[0]
        N, C = x.shape[1:]
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
        self.mlp = Mlp(in_features=dim,
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
        self.add_parameter("pos_embed", self.pos_embed)
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)
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

    def load_pretrained(self, path, rank=0, finetune=False):
        if not os.path.exists(path + '.pdparams'):
            raise ValueError("Model pretrain path {} does not "
                             "exists.".format(path))

        state_dict = self.state_dict()
        param_state_dict = paddle.load(path + ".pdparams")
        if not finetune:
            self.set_dict(param_state_dict)
            return

        for k in ['head0.weight', 'head0.bias', 'head.weight', 'head.bias']:
            if k in param_state_dict and param_state_dict[
                    k].shape != state_dict[k].shape:
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
        pos_tokens = paddle.nn.functional.interpolate(
            pos_tokens,
            size=(new_size, new_size),
            mode='bicubic',
            align_corners=False)
        pos_tokens = paddle.transpose(
            pos_tokens, perm=[0, 2, 3, 1]).flatten(1, 2)
        new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
        param_state_dict['pos_embed'] = new_pos_embed

        self.set_dict(param_state_dict)
        return

    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")


class ViTCELoss(nn.Layer):
    """
    ViT style Sigmoid Cross entropy loss
    """

    def __init__(self, epsilon=None):
        super().__init__()
        if epsilon is not None:
            assert epsilon >= 0 and epsilon <= 1, "epsilon must be in [0, 1]"
        self.epsilon = epsilon

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        class_num = x.shape[-1]
        if len(label.shape) == 1 or label.shape[-1] != class_num:
            label = F.one_hot(label, class_num)
            label = paddle.reshape(label, shape=[-1, class_num])
        if self.epsilon is not None:
            # vit style label smoothing
            with paddle.no_grad():
                label = label * (1.0 - self.epsilon) + self.epsilon

        if x.dtype == paddle.float16:
            x = paddle.cast(x, 'float32')
        loss = F.binary_cross_entropy_with_logits(x, label, reduction='none')
        loss = paddle.sum(loss, axis=-1)
        loss = loss.mean()

        return loss


class CELoss(nn.Layer):
    """
    Softmax Cross entropy loss
    """

    def __init__(self, epsilon=None):
        super().__init__()
        if epsilon is not None:
            assert epsilon >= 0 and epsilon <= 1, "epsilon must be in [0, 1]"
        self.epsilon = epsilon

    def _labelsmoothing(self, target, class_num):
        if len(target.shape) == 1 or target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._labelsmoothing(label, class_num)
            x = -F.log_softmax(x, axis=-1)
            loss = paddle.sum(x * label, axis=-1)
        else:
            if label.shape[-1] == x.shape[-1]:
                loss = paddle.sum(-label * F.log_softmax(x, axis=-1), axis=-1)
            else:
                if label.dtype == paddle.int32:
                    label = paddle.cast(label, 'int64')
                loss = F.cross_entropy(x, label=label, soft_label=False)
        loss = loss.mean()
        return loss


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
