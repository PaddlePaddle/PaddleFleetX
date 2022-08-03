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

from io import BytesIO
import math
import base64
import numpy as np
import paddle
import paddle.nn as nn
from visualdl import LogWriter
from PIL import Image


class VisualdlLogger(object):
    def __init__(self, log_dir):
        self.writer = LogWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step
                                   if step is None else step)

    def flush(self):
        self.writer.flush()


def tensor_to_base64(tensor):
    output_buffer = BytesIO()
    np.savetxt(output_buffer, tensor)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    base64_str = base64_str.decode('utf-8')
    return base64_str


def base64_to_tensor(b64_code):
    b64_decode = base64.b64decode(b64_code)
    return np.loadtxt(BytesIO(b64_decode))


def base64_to_image(base64_str):
    byte_data = base64.b64decode(base64_str)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    paddle.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def rearrange(tensor,
              pattern: str,
              b: int=-1,
              h: int=-1,
              w: int=-1,
              c: int=-1,
              x: int=-1,
              y: int=-1,
              n: int=-1):
    if pattern == 'b n (h d) -> b h n d':
        B, N, _ = tensor.shape
        return tensor.reshape([B, N, h, -1]).transpose([0, 2, 1, 3])
    elif pattern == 'b n (h d) -> (b h) n d':
        B, N, _ = tensor.shape
        return tensor.reshape([B, N, h, -1]).transpose([0, 2, 1, 3]).reshape(
            [B * h, N, -1])
    elif pattern == 'b (h c) x y -> (b h) (x y) c':
        B, _, _, _ = tensor.shape
        return tensor.reshape([B, h, -1, x, y]).transpose(
            [0, 1, 3, 4, 2]).reshape([B * h, x * y, -1])
    elif pattern == 'b n ... -> b n (...)':
        B, N = tensor.shape[:2]
        return tensor.reshape([B, N, -1])
    elif pattern == 'b j -> b 1 1 j':
        return tensor[:, None, None, :]
    elif pattern == 'b h n d -> b n (h d)':
        B, H, N, D = tensor.shape
        return tensor.transpose([0, 2, 1, 3]).reshape([B, N, -1])
    elif pattern == '(b h) (x y) d -> b (h d) x y':
        _, _, D = tensor.shape
        return tensor.reshape([-1, h, x, y, D]).transpose(
            [0, 1, 4, 2, 3]).reshape([-1, h * D, x, y])
    elif pattern == '(b h) n d -> b n (h d)':
        _, N, D = tensor.shape
        return tensor.reshape([-1, h, N, D]).transpose([0, 2, 1, 3]).reshape(
            [-1, N, h * D])
    elif pattern == 'b n -> b n 1':
        return tensor[:, :, None]
    elif pattern == 'b c h w -> b (h w) c':
        B, C, H, W = tensor.shape
        return tensor.transpose([0, 2, 3, 1]).reshape([B, -1, C])
    elif pattern == 'b (h w) c -> b c h w':
        B, _, C = tensor.shape
        return tensor.reshape([B, h, w, C]).transpose([0, 3, 1, 2])
    elif pattern == 'b (n d) -> b n d':
        B, _ = tensor.shape
        return tensor.reshape([B, n, -1])


def rearrange_many(tensors, pattern: str, h: int=-1, x: int=-1, y: int=-1):
    assert isinstance(tensors, (
        list, tuple)), "rearrange_many type must be list or tuple"
    if isinstance(tensors, tuple):
        tensors = list(tensors)
    if len(tensors) == 0:
        raise TypeError("Rearrange can't be applied to an empty list")
    for i, tensor in enumerate(tensors):
        tensors[i] = rearrange(tensor, pattern, h=h, x=x, y=y)
    return tensors


def repeat(tensor, pattern: str, h: int=-1, b: int=-1):
    if pattern == '1 -> b':
        return paddle.tile(tensor, repeat_times=b)
    elif pattern == 't -> b t':
        tensor = tensor[None, :]
        return paddle.tile(tensor, repeat_times=(b, 1))
    elif pattern == 'n d -> b n d':
        tensor = tensor[None, :]
        return paddle.tile(tensor, repeat_times=(b, 1, 1))
    elif pattern == 'o ... -> (o 4) ...':
        return paddle.tile(tensor, repeat_times=(4, 1, 1, 1))
    elif pattern == 'd -> b h 1 d':
        tensor = tensor[None, None, None, :]
        return paddle.tile(tensor, repeat_times=(b, h, 1, 1))
    elif pattern == 'd -> b 1 d':
        tensor = tensor[None, None, :]
        return paddle.tile(tensor, repeat_times=(b, 1, 1))


def repeat_many(tensors, pattern: str, h: int=-1, b: int=-1):
    assert isinstance(tensors, (list, tuple))
    if isinstance(tensors, tuple):
        tensors = list(tensors)
    if len(tensors) == 0:
        raise TypeError("Rearrange can't be applied to an empty list")
    for i, tensor in enumerate(tensors):
        tensors[i] = repeat(tensor, pattern, h=h, b=b)
    return tensors


def reduce(losses, pattern: str, reduction: str='mean'):
    if pattern == 'b ... -> b':
        axes = list(range(1, len(losses.shape)))
        return losses.mean(axes)


class EinopsToAndFrom(nn.Layer):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}',
                      **reconstitute_kwargs)
        return x


class Rearrange(nn.Layer):
    def __init__(self, pattern, n):
        super().__init__()
        self.pattern = pattern
        self.n = n

    def forward(self, x, **kwargs):
        shape = x.shape
        x = rearrange(x, f'{self.pattern}', n=self.n)
        return x
