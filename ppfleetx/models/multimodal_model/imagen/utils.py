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
from functools import partial, wraps

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import expm1

# helper functions


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(val, length=None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val, ) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output


def is_float_dtype(dtype):
    return any([
        dtype == float_dtype
        for float_dtype in (paddle.float64, paddle.float32, paddle.float16,
                            paddle.bfloat16)
    ])


def cast_uint8_images_to_float(images):
    if not images.dtype == paddle.uint8:
        return images
    return images / 255


zeros_ = nn.initializer.Constant(value=0.)


def zero_init_(m):
    zeros_(m.weight)
    if exists(m.bias):
        zeros_(m.bias)


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        if was_training:
            model.train(was_training)
        return out

    return inner


def pad_tuple_to_length(t, length, fillvalue=None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue, ) * remain_length))


# helper classes


class Identity(nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


# tensor helpers


def log(t, eps: float=1e-12):
    return paddle.log(t.clip(min=eps))


class Parallel(nn.Layer):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.LayerList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)


def l2norm(t):
    return F.normalize(t, axis=-1)


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.reshape([*t.shape, *((1, ) * padding_dims)])


def masked_mean(t, *, axis, mask=None):
    if not exists(mask):
        return t.mean(axis=axis)

    denom = mask.sum(axis=axis, keepdim=True)
    mask = mask[:, :, None]
    masked_t = paddle.where(mask == 0, paddle.to_tensor(0.), t)

    return masked_t.sum(axis=axis) / denom.clip(min=1e-5)


def resize_image_to(image, target_image_size, clamp_range=None):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(
        image, (target_image_size, target_image_size), mode='nearest')

    if exists(clamp_range):
        out = out.clip(*clamp_range)

    return out


# image normalization functions
# ddpms expect images to be in the range of -1 to 1


def normalize_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5


# classifier free guidance functions


def prob_mask_like(shape, prob):
    if prob == 1:
        return paddle.ones(shape, dtype=paddle.bool)
    elif prob == 0:
        return paddle.zeros(shape, dtype=paddle.bool)
    else:
        return paddle.zeros(shape).cast('float32').uniform_(0, 1) < prob


def rearrange(tensor,
              pattern: str,
              b: int=-1,
              h: int=-1,
              w: int=-1,
              c: int=-1,
              x: int=-1,
              y: int=-1,
              n: int=-1,
              s1: int=-1,
              s2: int=-1):
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
    elif pattern == 'b ... -> b (...)':
        B = tensor.shape[0]
        return tensor.reshape([B, -1])
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
    elif pattern == 'b ... -> b 1 ...':
        return tensor[:, None]
    elif pattern == 'b -> b 1 1 1':
        return tensor[:, None, None, None]
    elif pattern == 'b c (h s1) (w s2) -> b (c s1 s2) h w':
        assert s1 is not None
        assert s2 is not None
        B, C, H, W = tensor.shape
        tensor = tensor.reshape([B, C, H // s1, s1, W // s2, s2])
        tensor = tensor.transpose([0, 1, 3, 5, 2, 4])
        return tensor.reshape([B, C * s1 * s2, H // s1, W // s2])


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
        if b > 1:
            b = paddle.to_tensor(b)
            return paddle.tile(tensor, repeat_times=b)
        else:
            return tensor
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
    def __init__(self, pattern, n=None, s1=None, s2=None):
        super().__init__()
        self.pattern = pattern
        self.n = n
        self.s1 = s1
        self.s2 = s2

    def forward(self, x, **kwargs):
        x = rearrange(x, f'{self.pattern}', n=self.n, s1=self.s1, s2=self.s2)
        return x


# classifier free guidance functions

# gaussian diffusion with continuous time helper functions and classes
# large part of this was thanks to @crowsonkb at https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py


def beta_linear_log_snr(t):
    return -paddle.log(expm1(1e-4 + 10 * (t**2)))


def alpha_cosine_log_snr(t, s: float=0.008):
    return -log(
        (paddle.cos((t + s) / (1 + s) * math.pi * 0.5)**-2) - 1, eps=1e-5
    )  # not sure if this accounts for beta being clipped to 0.999 in discrete version


def log_snr_to_alpha_sigma(log_snr):
    return paddle.sqrt(F.sigmoid(log_snr)), paddle.sqrt(F.sigmoid(-log_snr))


class GaussianDiffusionContinuousTimes(nn.Layer):
    def __init__(self, *, noise_schedule, timesteps=1000):
        super().__init__()

        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.num_timesteps = timesteps

    def get_times(self, batch_size, noise_level):
        return paddle.full((batch_size, ), noise_level, dtype=paddle.float32)

    def sample_random_times(self, batch_size):
        return paddle.zeros((batch_size, )).cast('float32').uniform_(0, 1)

    def get_condition(self, times):
        return maybe(self.log_snr)(times)

    def get_sampling_timesteps(self, batch):
        times = paddle.linspace(1., 0., self.num_timesteps + 1)
        times = repeat(times, 't -> b t', b=batch)
        times = paddle.stack((times[:, :-1], times[:, 1:]), axis=0)
        times = times.unbind(axis=-1)
        return times

    def q_posterior(self, x_start, x_t, t, *, t_next=None):
        t_next = default(
            t_next, lambda: (t - 1. / self.num_timesteps).clip(min=0.))
        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(
            partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next**2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps=1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        dtype = x_start.dtype

        if isinstance(t, float):
            batch = x_start.shape[0]
            t = paddle.full((batch, ), t, dtype=dtype)

        noise = default(noise, lambda: paddle.randn(shape=x_start.shape, dtype=dtype))
        log_snr = self.log_snr(t).cast(dtype)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_padded_dim)

        return alpha * x_start + sigma * noise, log_snr, alpha, sigma

    def q_sample_from_to(self, x_from, from_t, to_t, noise=None):
        shape, dtype = x_from.shape, x_from.dtype
        batch = shape[0]

        if isinstance(from_t, float):
            from_t = paddle.full((batch, ), from_t, dtype=dtype)

        if isinstance(to_t, float):
            to_t = paddle.full((batch, ), to_t, dtype=dtype)

        noise = default(noise, lambda: paddle.randn(shape=x_from.shape, dtype=x_from.dtype))

        log_snr = self.log_snr(from_t)
        log_snr_padded_dim = right_pad_dims_to(x_from, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_padded_dim)

        log_snr_to = self.log_snr(to_t)
        log_snr_padded_dim_to = right_pad_dims_to(x_from, log_snr_to)
        alpha_to, sigma_to = log_snr_to_alpha_sigma(log_snr_padded_dim_to)

        return x_from * (alpha_to / alpha) + noise * (sigma_to * alpha - sigma
                                                      * alpha_to) / alpha

    def predict_start_from_v(self, x_t, t, v):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return alpha * x_t - sigma * v

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clip(min=1e-8)


class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val
