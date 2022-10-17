# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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


import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant, Normal

from .clip_modules import (
    constant_init,
    normal_init,
    Transformer,
    VisionTransformer,
    AttentionPool2D)


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2D(inplanes, planes, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)

        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.is_vd_mode = True if stride > 1 else False
        self.avgpool = nn.AvgPool2D(stride)  # if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2D(planes,
                               planes * self.expansion,
                               1,
                               bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                ("-1", nn.AvgPool2D(stride)),
                ("0",
                 nn.Conv2D(inplanes,
                           planes * self.expansion,
                           1,
                           stride=1,
                           bias_attr=False)),
                ("1", nn.BatchNorm2D(planes * self.expansion)))

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        if self.is_vd_mode:
            out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ModifiedResNet(nn.Layer):
    """
    A ResNet class that contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2D(3,
                               width // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width // 2)
        self.conv2 = nn.Conv2D(width // 2,
                               width // 2,
                               kernel_size=3,
                               padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(width // 2)
        self.conv3 = nn.Conv2D(width // 2,
                               width,
                               kernel_size=3,
                               padding=1,
                               bias_attr=False)
        self.bn3 = nn.BatchNorm2D(width)
        self.avgpool = nn.AvgPool2D(2)
        self.relu = nn.ReLU()

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2D(input_resolution // 32, embed_dim,
                                        heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                             (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.astype(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass Paddle's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.astype("float32"))
        return ret.astype(orig_type)


class QuickGELU(nn.Layer):

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class CLIP(nn.Layer):

    def __init__(self,
                 embed_dim=512,
                 # vision
                 image_resolution=224,
                 vision_layers=12,
                 vision_width=768,
                 vision_patch_size=32,
                 pre_norm=True,
                 proj=True,
                 patch_bias=False,
                 # text
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 qkv_bias=True,
                 use_recompute=False,
                 fused_linear=False):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                img_size=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                out_dim=embed_dim,
                depth=vision_layers,
                num_heads=vision_heads,
                pre_norm=pre_norm,
                proj=proj,
                patch_bias=patch_bias,
            )

        self.transformer = Transformer(
            embed_dim=transformer_width,
            depth=transformer_layers,
            num_heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = self.create_parameter(
            shape=(self.context_length, transformer_width),
            default_initializer=Normal(std=0.01))
        self.add_parameter("positional_embedding", self.positional_embedding)
        self.ln_final = LayerNorm(transformer_width)


        scale = transformer_width**-0.5
        self.text_projection = self.create_parameter(
            shape=(transformer_width, embed_dim),
            default_initializer=Normal(std=scale))
        self.add_parameter("text_projection", self.text_projection)

        logit_ = Constant(value=np.log(1 / 0.07))
        self.logit_scale = self.create_parameter(shape=(1, ),
                                                 default_initializer=logit_)
        self.add_parameter("logit_scale", self.logit_scale)

        self.initialize_parameters()

    def initialize_parameters(self):
        normal_init(self.token_embedding, std=0.02)
        normal_init(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.in_features**-0.5
                normal_init(self.visual.attnpool.q_proj, std=std)
                normal_init(self.visual.attnpool.k_proj, std=std)
                normal_init(self.visual.attnpool.v_proj, std=std)
                normal_init(self.visual.attnpool.out_proj, std=std)

            for resnet_block in [
                    self.visual.layer1, self.visual.layer2, self.visual.layer3,
                    self.visual.layer4
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        constant_init(param, 0)

        proj_std = (self.transformer.embed_dim**-0.5) * (
            (2 * self.transformer.depth))
        attn_std = self.transformer.embed_dim**-0.5
        fc_std = (2 * self.transformer.embed_dim)**-0.5
        for block in self.transformer.blocks:
            normal_init(block.attn.proj, std=proj_std)
            normal_init(block.attn.qkv, std=attn_std)
            normal_init(block.mlp.fc1, std=fc_std)
            normal_init(block.mlp.fc2, std=proj_std)

    def build_attention_mask(self):
        inf = paddle.to_tensor(float("-inf"))
        return paddle.tensor.triu((paddle.ones((
            self.context_length,
            self.context_length), dtype=paddle.get_default_dtype()) * inf), 1)

    @property
    def dtype(self):
        return self.visual.patch_embed.proj.weight.dtype

    def encode_image(self, image):
        return self.visual(image.astype(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).astype(
            self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.astype(self.dtype)
        x = self.transformer(x)
        x = self.ln_final(x).astype(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        idx = text.argmax(axis=-1)
        ran = paddle.arange(x.shape[0])
        x = paddle.concat([paddle.unsqueeze(x[i][idx[i]], axis=0) for i in ran],
                          axis=0)
        x = paddle.matmul(x, self.text_projection)

        return x

    def clip_logit_scale(self):
        self.logit_scale.clip(-4.6, 4.6)

    def forward(self, image, text, is_train=True):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(axis=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(axis=-1,
                                                           keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp() if is_train else 1
        image_logits = paddle.matmul(logit_scale * image_features,
                                     text_features.t())
        text_logits = paddle.matmul(logit_scale * text_features,
                                    image_features.t())
        self.clip_logit_scale()

        return image_logits, text_logits


def convert_weights(model):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1D, nn.Conv2D, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiHeadAttention):
            for attr in [
                    *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                    "in_proj_bias", "bias_k", "bias_v"
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def clip_vit_base_32(**kwargs):
    model = CLIP(embed_dim=512,
                 image_resolution=224,
                 vision_layers=12,
                 vision_width=768,
                 vision_patch_size=32,
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 qkv_bias=True,
                 pre_norm=True,
                 proj=True,
                 patch_bias=False,
                 **kwargs)
    return model


def clip_vit_base_16(**kwargs):
    model = CLIP(embed_dim=512,
                 image_resolution=224,
                 vision_layers=12,
                 vision_width=768,
                 vision_patch_size=16,
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 qkv_bias=True,
                 pre_norm=True,
                 proj=True,
                 patch_bias=False,
                 **kwargs)
    return model


def clip_vit_large_14(**kwargs):
    model = CLIP(embed_dim=768,
                 image_resolution=224,
                 vision_layers=24,
                 vision_width=1024,
                 vision_patch_size=14,
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=768,
                 transformer_heads=12,
                 transformer_layers=12,
                 qkv_bias=True,
                 pre_norm=True,
                 proj=True,
                 patch_bias=False,
                 **kwargs)
    return model



def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.patch_embed.proj.weight"].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.qkv.weight")
        ])
        vision_patch_size = state_dict["visual.patch_embed.proj.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[1] - 1)**0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2] for k in state_dict
                    if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] -
             1)**0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict[
            "visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict
            if k.startswith(f"transformer.blocks")))

    model = CLIP(embed_dim=embed_dim,
                 image_resolution=image_resolution,
                 vision_layers=vision_layers,
                 vision_width=vision_width,
                 vision_patch_size=vision_patch_size,
                 context_length=context_length,
                 vocab_size=vocab_size,
                 transformer_width=transformer_width,
                 transformer_heads=transformer_heads,
                 transformer_layers=transformer_layers)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


class CLIPCriterion(nn.Layer):
    """
    Criterion for CLIP. It calculates the final loss.
    """
    def __init__(self):
        super(CLIPCriterion, self).__init__()
        self.criterion= nn.CrossEntropyLoss()
    
    def forward(self, img_logits, text_logits, img_labels, text_labels):
        img_loss = self.criterion(img_logits, img_labels)
        text_loss = self.criterion(text_logits, text_labels)
        loss = img_loss + text_loss
        return loss 
