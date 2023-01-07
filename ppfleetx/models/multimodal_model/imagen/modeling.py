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

from tqdm import tqdm
from functools import partial
from contextlib import contextmanager, nullcontext

import paddle
import paddle.nn.functional as F
from paddle import nn
import paddle.vision.transforms as T

from .unet import Unet
from ppfleetx.models.language_model.t5 import *
from ppfleetx.data.tokenizers import get_t5_tokenizer
from .utils import (
    GaussianDiffusionContinuousTimes, default, exists, cast_tuple, first,
    maybe, eval_decorator, identity, pad_tuple_to_length, right_pad_dims_to,
    resize_image_to, normalize_neg_one_to_one, rearrange, repeat, reduce,
    unnormalize_zero_to_one, cast_uint8_images_to_float, is_float_dtype)


# predefined unets, with configs lining up with hyperparameters in appendix of paper
class Unet64_397M(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim=256,
            dim_mults=(1, 2, 3, 4),
            num_resnet_blocks=3,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True),
            attn_heads=8,
            ff_mult=2.,
            memory_efficient=False)
        super().__init__(*args, **{ ** default_kwargs, ** kwargs})


class BaseUnet64(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim=512,
            cond_dim=512,
            dim_mults=(1, 2, 3, 4),
            num_resnet_blocks=3,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True),
            attn_heads=8,
            ff_mult=2.,
            memory_efficient=False)
        super().__init__(*args, **{ ** default_kwargs, ** kwargs})


class SRUnet256(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, False, True),
            attn_heads=8,
            ff_mult=2.,
            memory_efficient=True)
        super().__init__(*args, **{ ** default_kwargs, ** kwargs})


class SRUnet1024(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=False,
            layer_cross_attns=(False, False, False, True),
            attn_heads=8,
            ff_mult=2.,
            memory_efficient=True)
        super().__init__(*args, **{ ** default_kwargs, ** kwargs})


# main imagen ddpm class, which is a cascading DDPM from Ho et al.
class ImagenCriterion(nn.Layer):
    """
    Criterion for Imagen. It calculates the final loss.
    """

    def __init__(self, name='mse_loss', p2_loss_weight_k=1):
        super(ImagenCriterion, self).__init__()
        self.p2_loss_weight_k = p2_loss_weight_k

        if name == 'l1_loss':
            self.loss_func = F.l1_loss
        elif name == 'mse_loss':
            self.loss_func = F.mse_loss
        elif name == 'smooth_l1_loss':
            self.loss_func = F.smooth_l1_loss
        else:
            raise NotImplementedError()

    def forward(self, pred, target, log_snr, p2_loss_weight_gamma):
        """
        Args:
            pred(Tensor):
                The logits of prediction. Its data type should be float32 and
                its shape is [batch_size, sequence_length, vocab_size].
            target(Tensor):
                The labels of the prediction, default is noise.

        Returns:
            Tensor: The pretraining loss. Its data type should be float32 and its shape is [1].

        """
        losses = self.loss_func(pred, target, reduction="none")
        losses = reduce(losses, 'b ... -> b', 'mean')

        # p2 loss reweighting

        if p2_loss_weight_gamma > 0:
            loss_weight = (
                self.p2_loss_weight_k + log_snr.exp())**-p2_loss_weight_gamma
            losses = losses * loss_weight

        return losses.mean()


class ImagenModel(nn.Layer):
    def __init__(
            self,
            unets,
            image_sizes,
            text_encoder_name=None,
            text_embed_dim=1024,
            channels=3,
            timesteps=1000,
            cond_drop_prob=0.1,
            noise_schedules='cosine',
            pred_objectives='noise',
            random_crop_sizes=None,
            lowres_noise_schedule='linear',
            lowres_sample_noise_level=0.2,
            per_sample_random_aug_noise_level=False,
            condition_on_text=True,
            auto_normalize_img=True,
            p2_loss_weight_gamma=0.5,
            dynamic_thresholding=True,
            dynamic_thresholding_percentile=0.95,
            only_train_unet_number=None,
            is_sr=False,
            is_video=False,
            fused_linear=False, ):
        super().__init__()

        # conditioning hparams

        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text
        self.is_sr = is_sr
        self.is_video = is_video

        # channels

        self.channels = channels

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        unets = cast_tuple(unets)
        num_unets = len(unets)

        # determine noise schedules per unet

        timesteps = cast_tuple(timesteps, num_unets)

        # make sure noise schedule defaults to 'cosine', 'cosine', and then 'linear' for rest of super-resoluting unets

        noise_schedules = cast_tuple(noise_schedules)
        noise_schedules = pad_tuple_to_length(noise_schedules, 2, 'cosine')
        noise_schedules = pad_tuple_to_length(noise_schedules, num_unets,
                                              'linear')

        # construct noise schedulers

        noise_scheduler_klass = GaussianDiffusionContinuousTimes
        self.noise_schedulers = nn.LayerList([])

        for timestep, noise_schedule in zip(timesteps, noise_schedules):
            noise_scheduler = noise_scheduler_klass(
                noise_schedule=noise_schedule, timesteps=timestep)
            self.noise_schedulers.append(noise_scheduler)

        # randomly cropping for upsampler training

        self.random_crop_sizes = cast_tuple(random_crop_sizes, num_unets)
        assert not exists(
            first(self.random_crop_sizes)
        ), 'you should not need to randomly crop image during training for base unet, only for upsamplers - so pass in `random_crop_sizes = (None, 128, 256)` as example'

        # lowres augmentation noise schedule

        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(
            noise_schedule=lowres_noise_schedule)

        # ddpm objectives - predicting noise by default

        self.pred_objectives = cast_tuple(pred_objectives, num_unets)

        # get text encoder

        self.text_encoder_name = text_encoder_name

        if text_encoder_name is None:
            pass
        elif 't5' in text_encoder_name:
            self.text_embed_dim = default(
                text_embed_dim, lambda: get_encoded_dim(text_encoder_name))
            self.t5_encoder = get_t5_model(
                name=text_encoder_name, pretrained=True)
            self.tokenizer = get_t5_tokenizer(name=text_encoder_name)
            self.t5_encode_text = t5_encode_text
        elif 'deberta' in text_encoder_name:
            self.text_embed_dim = default(
                text_embed_dim,
                lambda: get_debertav2_encoded_dim(text_encoder_name))
            self.debertav2_encoder = get_debertav2_model(
                name=text_encoder_name, pretrained=True)
            self.tokenizer = get_debertav2_tokenizer(name=text_encoder_name)
            self.debertav2_encode_text = debertav2_encode_text
        else:
            raise NotImplementedError("Please implement the text encoder.")

        # construct unets

        self.unets = nn.LayerList([])

        self.unet_being_trained_index = -1  # keeps track of which unet is being trained at the moment
        self.only_train_unet_number = only_train_unet_number

        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, Unet)
            is_first = ind == 0

            one_unet = one_unet.cast_model_parameters(
                cond_on_text=self.condition_on_text,
                text_embed_dim=self.text_embed_dim
                if self.condition_on_text else None,
                channels=self.channels,
                channels_out=self.channels)

            self.unets.append(one_unet)

        # unet image sizes

        image_sizes = cast_tuple(image_sizes)
        self.image_sizes = image_sizes

        self.sample_channels = cast_tuple(self.channels, num_unets)

        self.right_pad_dims_to_datatype = partial(
            rearrange, pattern=('b -> b 1 1 1'))

        # cascading ddpm related stuff

        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level

        # classifier free guidance

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)

        # dynamic thresholding

        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # p2 loss weight

        self.p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        assert all([
            (gamma_value <= 2) for gamma_value in self.p2_loss_weight_gamma
        ]), 'in paper, they noticed any gamma greater than 2 is harmful'

        # one temp parameter for keeping track of device

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.LayerList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list
        self.unet_being_trained_index = index
        return self.unets[index]

    def reset_unets(self, ):
        self.unets = nn.LayerList([*self.unets])
        self.unet_being_trained_index = -1

    @contextmanager
    def one_unet_in_gpu(self, unet_number=None, unet=None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        yield

    def reset_unets_all(self, ):
        self.unets = nn.LayerList([*self.unets])
        self.unet_being_trained_index = -1

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_unets()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_unets_all()
        return self.unets[self.unet_being_trained_index].set_state_dict(
            *args, **kwargs)

    # gaussian diffusion methods

    def p_mean_variance(self,
                        unet,
                        x,
                        t,
                        *,
                        noise_scheduler,
                        text_embeds=None,
                        text_mask=None,
                        cond_images=None,
                        lowres_cond_img=None,
                        self_cond=None,
                        lowres_noise_times=None,
                        cond_scale=1.,
                        model_output=None,
                        t_next=None,
                        pred_objective='noise',
                        dynamic_threshold=True):
        assert not (
            cond_scale != 1. and not self.can_classifier_guidance
        ), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'
        time_var = noise_scheduler.get_condition(t)
        pred = default(model_output, lambda: unet.forward_with_cond_scale(x, time_var, text_embeds = text_embeds, text_mask = text_mask, cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_noise_times)))

        if pred_objective == 'noise':
            x_start = noise_scheduler.predict_start_from_noise(
                x, t=t, noise=pred)
        elif pred_objective == 'x_start':
            x_start = pred
        elif pred_objective == 'v':
            x_start = noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        if dynamic_threshold:
            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
            s = paddle.quantile(
                rearrange(x_start, 'b ... -> b (...)').abs(),
                self.dynamic_thresholding_percentile,
                axis=-1)

            s.clip_(min=1.)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clip(-s, s) / s
        else:
            x_start.clip_(-1., 1.)

        mean_and_variance = noise_scheduler.q_posterior(
            x_start=x_start, x_t=x, t=t, t_next=t_next)
        return mean_and_variance, x_start

    @paddle.no_grad()
    def p_sample(self,
                 unet,
                 x,
                 t,
                 *,
                 noise_scheduler,
                 t_next=None,
                 text_embeds=None,
                 text_mask=None,
                 cond_images=None,
                 cond_scale=1.,
                 self_cond=None,
                 lowres_cond_img=None,
                 lowres_noise_times=None,
                 pred_objective='noise',
                 dynamic_threshold=True):
        b = x.shape[0]
        (model_mean, _, model_log_variance), x_start = self.p_mean_variance(
            unet,
            x=x,
            t=t,
            t_next=t_next,
            noise_scheduler=noise_scheduler,
            text_embeds=text_embeds,
            text_mask=text_mask,
            cond_images=cond_images,
            cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img,
            self_cond=self_cond,
            lowres_noise_times=lowres_noise_times,
            pred_objective=pred_objective,
            dynamic_threshold=dynamic_threshold)
        noise = paddle.randn(shape=x.shape, dtype=x.dtype)
        # no noise when t == 0
        is_last_sampling_timestep = (t_next == 0) if isinstance(
            noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.cast('float32')).reshape(
            [b, *((1, ) * (len(x.shape) - 1))])
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance
                                            ).exp() * noise
        return pred, x_start

    @paddle.no_grad()
    def p_sample_loop(self,
                      unet,
                      shape,
                      *,
                      noise_scheduler,
                      lowres_cond_img=None,
                      lowres_noise_times=None,
                      text_embeds=None,
                      text_mask=None,
                      cond_images=None,
                      inpaint_images=None,
                      inpaint_masks=None,
                      inpaint_resample_times=5,
                      init_images=None,
                      skip_steps=None,
                      cond_scale=1,
                      pred_objective='noise',
                      dynamic_threshold=True):

        batch = shape[0]
        img = paddle.randn(shape)

        # for initialization with an image or video

        if exists(init_images):
            img += init_images

        # keep track of x0, for self conditioning

        x_start = None

        # prepare inpainting

        has_inpainting = exists(inpaint_images) and exists(inpaint_masks)
        resample_times = inpaint_resample_times if has_inpainting else 1

        if has_inpainting:
            inpaint_images = self.normalize_img(inpaint_images)
            inpaint_images = resize_image_to(inpaint_images, shape[-1])
            inpaint_masks = resize_image_to(
                rearrange(inpaint_masks, 'b ... -> b 1 ...').cast('float32'),
                shape[-1]).cast('bool')

        # time

        timesteps = noise_scheduler.get_sampling_timesteps(batch)

        # whether to skip any steps

        skip_steps = default(skip_steps, 0)
        timesteps = timesteps[skip_steps:]

        for times, times_next in tqdm(
                timesteps, desc='sampling loop time step',
                total=len(timesteps)):
            is_last_timestep = times_next == 0

            for r in reversed(range(resample_times)):
                is_last_resample_step = r == 0

                if has_inpainting:
                    noised_inpaint_images, *_ = noise_scheduler.q_sample(
                        inpaint_images, t=times)
                    img = img * ~inpaint_masks + noised_inpaint_images * inpaint_masks

                self_cond = x_start if unet.self_cond else None

                img, x_start = self.p_sample(
                    unet,
                    img,
                    times,
                    t_next=times_next,
                    text_embeds=text_embeds,
                    text_mask=text_mask,
                    cond_images=cond_images,
                    cond_scale=cond_scale,
                    self_cond=self_cond,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                    pred_objective=pred_objective,
                    dynamic_threshold=dynamic_threshold)

                if has_inpainting and not (is_last_resample_step or
                                           paddle.all(is_last_timestep)):
                    renoised_img = noise_scheduler.q_sample_from_to(
                        img, times_next, times)

                    img = paddle.where(
                        self.right_pad_dims_to_datatype(is_last_timestep), img,
                        renoised_img)

        img.clip_(-1., 1.)

        # final inpainting

        if has_inpainting:
            img = img * ~inpaint_masks + inpaint_images * inpaint_masks

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @paddle.no_grad()
    @eval_decorator
    def sample(
            self,
            texts=None,
            text_masks=None,
            text_embeds=None,
            cond_images=None,
            inpaint_images=None,
            inpaint_masks=None,
            inpaint_resample_times=5,
            init_images=None,
            skip_steps=None,
            batch_size=1,
            cond_scale=1.,
            lowres_sample_noise_level=None,
            start_at_unet_number=1,
            start_image_or_video=None,
            stop_at_unet_number=None,
            return_all_unet_outputs=True,
            return_pil_images=False, ):
        self.reset_unets()

        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        if exists(texts) and not exists(
                text_embeds) and not self.unconditional:
            with paddle.amp.auto_cast(enable=False):
                if 't5' in self.text_encoder_name:
                    text_embeds, text_masks = self.t5_encode_text(
                        t5=self.t5_encoder, texts=texts, return_attn_mask=True)
                elif 'debert' in self.text_encoder_name:
                    text_embeds, text_masks = self.debertav2_encode_text(
                        debertav2=self.debertav2_encoder,
                        texts=texts,
                        return_attn_mask=True)

        if not self.unconditional:
            text_masks = default(
                text_masks, lambda: paddle.any(text_embeds != 0., axis=-1))
            batch_size = text_embeds.shape[0]

        if exists(inpaint_images):
            if self.unconditional:
                if batch_size == 1:  # assume researcher wants to broadcast along inpainted images
                    batch_size = inpaint_images.shape[0]

            assert inpaint_images.shape[
                0] == batch_size, 'number of inpainting images must be equal to the specified batch size on sample `sample(batch_size=<int>)``'
            assert not (
                self.condition_on_text and
                inpaint_images.shape[0] != text_embeds.shape[0]
            ), 'number of inpainting images must be equal to the number of text to be conditioned on'

        assert not (
            self.condition_on_text and not exists(text_embeds)
        ), 'text or text encodings must be passed into imagen if specified'
        assert not (
            not self.condition_on_text and exists(text_embeds)
        ), 'imagen specified not to be conditioned on text, yet it is presented'
        assert not (
            exists(text_embeds) and
            text_embeds.shape[-1] != self.text_embed_dim
        ), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        assert not (
            exists(inpaint_images) ^ exists(inpaint_masks)
        ), 'inpaint images and masks must be both passed in to do inpainting'

        outputs = []

        lowres_sample_noise_level = default(lowres_sample_noise_level,
                                            self.lowres_sample_noise_level)

        num_unets = len(self.unets)

        # condition scaling

        cond_scale = cast_tuple(cond_scale, num_unets)

        # for initial image and skipping steps

        init_images = cast_tuple(init_images, num_unets)
        init_images = [
            maybe(self.normalize_img)(init_image) for init_image in init_images
        ]

        skip_steps = cast_tuple(skip_steps, num_unets)

        # handle starting at a unet greater than 1, for training only-upscaler training

        if start_at_unet_number > 1:

            assert not exists(stop_at_unet_number
                              ) or start_at_unet_number <= stop_at_unet_number
            assert exists(
                start_image_or_video
            ), 'starting image or video must be supplied if only doing upscaling'

            prev_image_size = self.image_sizes[start_at_unet_number - 1]
            img = resize_image_to(start_image_or_video, prev_image_size)

        # go through each unet in cascade

        for unet_number, unet, channel, image_size, noise_scheduler, pred_objective, dynamic_threshold, unet_cond_scale, unet_init_images, unet_skip_steps in tqdm(
                zip(
                    range(1, num_unets + 1), self.unets, self.sample_channels,
                    self.image_sizes, self.noise_schedulers,
                    self.pred_objectives, self.dynamic_thresholding,
                    cond_scale, init_images, skip_steps)):

            lowres_cond_img = lowres_noise_times = None
            shape = (batch_size, channel, image_size, image_size)

            if unet.lowres_cond:
                lowres_noise_times = self.lowres_noise_schedule.get_times(
                    batch_size, lowres_sample_noise_level)

                lowres_cond_img = resize_image_to(img, image_size)
                lowres_cond_img = self.normalize_img(lowres_cond_img)
                lowres_cond_img, *_ = self.lowres_noise_schedule.q_sample(
                    x_start=lowres_cond_img,
                    t=lowres_noise_times,
                    noise=paddle.randn(
                        shape=lowres_cond_img.shape,
                        dtype=lowres_cond_img.dtype))

            if exists(unet_init_images):
                unet_init_images = resize_image_to(unet_init_images,
                                                   image_size)

            shape = (batch_size, self.channels, image_size, image_size)

            img = self.p_sample_loop(
                unet,
                shape,
                text_embeds=text_embeds,
                text_mask=text_masks,
                cond_images=cond_images,
                inpaint_images=inpaint_images,
                inpaint_masks=inpaint_masks,
                inpaint_resample_times=inpaint_resample_times,
                init_images=unet_init_images,
                skip_steps=unet_skip_steps,
                cond_scale=unet_cond_scale,
                lowres_cond_img=lowres_cond_img,
                lowres_noise_times=lowres_noise_times,
                noise_scheduler=noise_scheduler,
                pred_objective=pred_objective,
                dynamic_threshold=dynamic_threshold)

            outputs.append(img)

            if exists(stop_at_unet_number
                      ) and stop_at_unet_number == unet_number:
                break

        output_index = -1 if not return_all_unet_outputs else slice(
            None)  # either return last unet output or all unet outputs

        if not return_pil_images:
            return outputs[output_index]

        if not return_all_unet_outputs:
            outputs = outputs[-1:]

        pil_images = list(
            map(lambda img: list(map(T.ToPILImage(), img.unbind(dim=0))),
                outputs))

        return pil_images[
            output_index]  # now you have a bunch of pillow images you can just .save(/where/ever/you/want.png)

    def p_losses(self,
                 unet,
                 x_start,
                 times,
                 *,
                 noise_scheduler,
                 lowres_cond_img=None,
                 lowres_aug_times=None,
                 text_embeds=None,
                 text_mask=None,
                 cond_images=None,
                 noise=None,
                 times_next=None,
                 pred_objective='noise',
                 p2_loss_weight_gamma=0.,
                 random_crop_size=None):
        is_video = x_start.ndim == 5

        noise = default(noise, lambda: paddle.randn(shape=x_start.shape, dtype=x_start.dtype))

        # normalize to [-1, 1]

        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # random cropping during training
        # for upsamplers

        if exists(random_crop_size):
            if is_video:
                frames = x_start.shape[2]
                x_start, lowres_cond_img, noise = rearrange_many(
                    (x_start, lowres_cond_img,
                     noise), 'b c f h w -> (b f) c h w')

            aug = K.RandomCrop((random_crop_size, random_crop_size), p=1.)

            # make sure low res conditioner and image both get augmented the same way
            # detailed https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            x_start = aug(x_start)
            lowres_cond_img = aug(lowres_cond_img, params=aug._params)
            noise = aug(noise, params=aug._params)

            if is_video:
                x_start, lowres_cond_img, noise = rearrange_many(
                    (x_start, lowres_cond_img, noise),
                    '(b f) c h w -> b c f h w',
                    f=frames)

        # get x_t

        x_noisy, log_snr, alpha, sigma = noise_scheduler.q_sample(
            x_start=x_start, t=times, noise=noise)

        # also noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy, *_ = self.lowres_noise_schedule.q_sample(
                x_start=lowres_cond_img,
                t=lowres_aug_times,
                noise=paddle.randn(
                    shape=lowres_cond_img.shape, dtype=lowres_cond_img.dtype))

        # time condition

        noise_cond = noise_scheduler.get_condition(times)

        # unet kwargs

        unet_kwargs = dict(
            text_embeds=text_embeds,
            text_mask=text_mask,
            cond_images=cond_images,
            lowres_noise_times=self.lowres_noise_schedule.get_condition(
                lowres_aug_times),
            lowres_cond_img=lowres_cond_img_noisy,
            cond_drop_prob=self.cond_drop_prob, )

        # self condition if needed

        # Because 'unet' can be an instance of DistributedDataParallel coming from the
        # ImagenTrainer.unet_being_trained when invoking ImagenTrainer.forward(), we need to
        # access the member 'module' of the wrapped unet instance.
        self_cond = unet._layers.self_cond if isinstance(
            unet, paddle.DataParallel) else unet.self_cond

        if self_cond and random() < 0.5:
            with paddle.no_grad():
                pred = unet.forward(x_noisy, noise_cond,
                                    **unet_kwargs).detach()

                x_start = noise_scheduler.predict_start_from_noise(
                    x_noisy, t=times,
                    noise=pred) if pred_objective == 'noise' else pred

                unet_kwargs = { ** unet_kwargs, 'self_cond': x_start}

        # get prediction

        pred = unet.forward(x_noisy, noise_cond, **unet_kwargs)

        # prediction objective

        if pred_objective == 'noise':
            target = noise
        elif pred_objective == 'x_start':
            target = x_start
        elif pred_objective == 'v':
            # derivation detailed in Appendix D of Progressive Distillation paper
            # https://arxiv.org/abs/2202.00512
            # this makes distillation viable as well as solve an issue with color shifting in upresoluting unets, noted in imagen-video
            target = alpha * noise - sigma * x_start
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        return pred, target, log_snr, p2_loss_weight_gamma

    def forward(self,
                images,
                unet=None,
                texts=None,
                text_embeds=None,
                text_masks=None,
                unet_number=None,
                cond_images=None):
        if self.is_video and images.ndim == 4:
            images = rearrange(images, 'b c h w -> b c 1 h w')

        assert images.shape[-1] == images.shape[
            -2], f'the images you pass in must be a square, but received dimensions of {images.shape[2]}, {images.shape[-1]}'
        assert not (
            len(self.unets) > 1 and not exists(unet_number)
        ), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        assert not exists(
            self.only_train_unet_number
        ) or self.only_train_unet_number == unet_number, 'you can only train on unet #{self.only_train_unet_number}'

        images = cast_uint8_images_to_float(images)
        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        assert is_float_dtype(
            images.dtype
        ), f'images tensor needs to be floats but {images.dtype} dtype found instead'

        unet_index = unet_number - 1

        unet = default(unet, lambda: self.get_unet(unet_number))

        noise_scheduler = self.noise_schedulers[unet_index]
        p2_loss_weight_gamma = self.p2_loss_weight_gamma[unet_index]
        pred_objective = self.pred_objectives[unet_index]
        target_image_size = self.image_sizes[unet_index]
        random_crop_size = self.random_crop_sizes[unet_index]
        if self.is_sr:
            prev_image_size = self.image_sizes[unet_index - 1]
        else:
            prev_image_size = None
        b, c, h, w = images.shape

        assert images.shape[1] == self.channels
        assert h >= target_image_size and w >= target_image_size

        times = noise_scheduler.sample_random_times(b)

        if exists(texts) and not exists(
                text_embeds) and not self.unconditional:
            assert len(texts) == len(
                images
            ), 'number of text captions does not match up with the number of images given'
            with paddle.amp.auto_cast(enable=False):
                if 't5' in self.text_encoder_name:
                    text_embeds, text_masks = self.t5_encode_text(
                        t5=self.t5_encoder,
                        texts=texts,
                        tokenizer=self.tokenizer,
                        return_attn_mask=True)
                elif 'deberta' in self.text_encoder_name:
                    text_embeds, text_masks = self.debertav2_encode_text(
                        debertav2=self.debertav2_encoder,
                        texts=texts,
                        tokenizer=self.tokenizer,
                        return_attn_mask=True)
                else:
                    raise NotImplementedError(
                        "Please implement the text encoder.")

        if not self.unconditional:
            text_masks = default(
                text_masks, lambda: paddle.any(text_embeds != 0., axis=-1))

        assert not (
            self.condition_on_text and not exists(text_embeds)
        ), 'text or text encodings must be passed into decoder if specified'
        assert not (
            not self.condition_on_text and exists(text_embeds)
        ), 'decoder specified not to be conditioned on text, yet it is presented'

        assert not (
            exists(text_embeds) and
            text_embeds.shape[-1] != self.text_embed_dim
        ), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = resize_image_to(images, prev_image_size)
            lowres_cond_img = resize_image_to(lowres_cond_img,
                                              target_image_size)

            if self.per_sample_random_aug_noise_level:
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(
                    b)
            else:
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(
                    1)
                lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b=b)

        images = resize_image_to(images, target_image_size)

        return self.p_losses(
            unet,
            images,
            times,
            text_embeds=text_embeds,
            text_mask=text_masks,
            cond_images=cond_images,
            noise_scheduler=noise_scheduler,
            lowres_cond_img=lowres_cond_img,
            lowres_aug_times=lowres_aug_times,
            pred_objective=pred_objective,
            p2_loss_weight_gamma=p2_loss_weight_gamma,
            random_crop_size=random_crop_size)


def imagen_397M_text2im_64(**kwargs):
    use_recompute = kwargs.pop('use_recompute')
    recompute_granularity = kwargs.pop('recompute_granularity')
    model = ImagenModel(
        unets=Unet64_397M(use_recompute=use_recompute),
        image_sizes=(64, ),
        **kwargs)
    return model


def imagen_text2im_64(**kwargs):
    use_recompute = kwargs.pop('use_recompute')
    recompute_granularity = kwargs.pop('recompute_granularity')
    if 'lowres_cond' in kwargs:
        lowres_cond = kwargs.pop('lowres_cond')
    else:
        lowres_cond = False
    model = ImagenModel(
        unets=BaseUnet64(
            lowres_cond=lowres_cond, use_recompute=use_recompute),
        image_sizes=(64, ),
        **kwargs)
    return model


def imagen_text2im_64_debertav2(**kwargs):
    use_recompute = kwargs.pop('use_recompute')
    recompute_granularity = kwargs.pop('recompute_granularity')
    model = ImagenModel(
        unets=BaseUnet64(
            dim=360, use_recompute=use_recompute),
        image_sizes=(64, ),
        **kwargs)
    return model


def imagen_text2im_64_SR256(**kwargs):
    use_recompute = kwargs.pop('use_recompute')
    recompute_granularity = kwargs.pop('recompute_granularity')
    model = ImagenModel(
        unets=(BaseUnet64(use_recompute=use_recompute),
               SRUnet256(use_recompute=use_recompute)),
        image_sizes=(64, 256),
        **kwargs)
    return model


def imagen_SR256(**kwargs):
    use_recompute = kwargs.pop('use_recompute')
    recompute_granularity = kwargs.pop('recompute_granularity')
    if 'lowres_cond' in kwargs:
        lowres_cond = kwargs.pop('lowres_cond')
    else:
        lowres_cond = False
    model = ImagenModel(
        unets=SRUnet256(
            lowres_cond=lowres_cond, use_recompute=use_recompute),
        image_sizes=(256, 64),
        **kwargs)
    return model


def imagen_SR1024(**kwargs):
    use_recompute = kwargs.pop('use_recompute')
    recompute_granularity = kwargs.pop('recompute_granularity')
    if 'lowres_cond' in kwargs:
        lowres_cond = kwargs.pop('lowres_cond')
    else:
        lowres_cond = False
    model = ImagenModel(
        unets=SRUnet1024(
            dim=128, lowres_cond=lowres_cond, use_recompute=use_recompute),
        image_sizes=(1024, 256),
        **kwargs)
    return model
