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

from typing import List
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager, nullcontext

import paddle
import paddle.nn.functional as F
from paddle import nn
import paddle.vision.transforms as T

from fleetx.models.imagen_model.modeling_imagen_modules import (
    GaussianDiffusion,
    GaussianDiffusionContinuousTimes,
    Unet,
    eval_decorator,
    exists,
    identity,
    cast_tuple,
    maybe,
    default,
    pad_tuple_to_length,
    right_pad_dims_to,
    resize_image_to,
    normalize_neg_one_to_one,
    unnormalize_zero_to_one,
    rearrange,
    repeat,
    reduce, )


# predefined unets, with configs lining up with hyperparameters in appendix of paper
class Unet64_300M(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim=224,
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


class Imagen(nn.Layer):
    def __init__(
            self,
            unets,
            image_sizes,  # for cascading ddpm, image size at each stage
            text_encoder_name='t5/t5-11b',
            text_embed_dim=1024,
            in_chans=3,
            timesteps=1000,
            cond_drop_prob=0.1,
            loss_type='l2',
            num_classes=None,
            noise_schedules='cosine',
            pred_objectives='noise',
            lowres_noise_schedule='linear',
            lowres_sample_noise_level=0.2,  # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
            per_sample_random_aug_noise_level=False,  # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
            condition_on_text=True,
            auto_normalize_img=True,  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
            continuous_times=True,
            p2_loss_weight_gamma=0.5,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time
            p2_loss_weight_k=1,
            dynamic_thresholding=True,
            dynamic_thresholding_percentile=0.9,  # unsure what this was based on perusal of paper
            only_train_unet_number=None, ):
        super().__init__()

        # deepspeed        

        # loss

        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # conditioning hparams

        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text

        # channels

        self.channels = in_chans

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

        noise_scheduler_klass = GaussianDiffusion if not continuous_times else GaussianDiffusionContinuousTimes
        self.noise_schedulers = nn.LayerList([])

        for timestep, noise_schedule in zip(timesteps, noise_schedules):
            noise_scheduler = noise_scheduler_klass(
                noise_schedule=noise_schedule, timesteps=timestep)
            self.noise_schedulers.append(noise_scheduler)

        # lowres augmentation noise schedule

        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(
            noise_schedule=lowres_noise_schedule)

        # ddpm objectives - predicting noise by default

        self.pred_objectives = cast_tuple(pred_objectives, num_unets)

        # get text encoder

        self.text_encoder_name = text_encoder_name
        #self.text_embed_dim = default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))
        self.text_embed_dim = default(text_embed_dim, lambda: 1024)

        # construct unets

        self.unets = nn.LayerList([])

        self.unet_being_trained_index = -1  # keeps track of which unet is being trained at the moment
        self.only_train_unet_number = only_train_unet_number

        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, Unet)
            is_first = ind == 0

            one_unet = one_unet.cast_model_parameters(
                lowres_cond=not is_first,
                text_embed_dim=self.text_embed_dim
                if self.condition_on_text else None,
                channels=self.channels,
                channels_out=self.channels,
                cond_on_text=self.condition_on_text,
                learned_sinu_pos_emb=continuous_times)

            self.unets.append(one_unet)

        # unet image sizes

        self.image_sizes = cast_tuple(image_sizes)
        assert num_unets == len(image_sizes)

        self.sample_channels = cast_tuple(self.channels, num_unets)

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (
            False, *((True, ) * (num_unets - 1))
        ), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

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

        self.p2_loss_weight_k = p2_loss_weight_k
        self.p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        assert all([
            (gamma_value <= 2) for gamma_value in self.p2_loss_weight_gamma
        ]), 'in paper, they noticed any gamma greater than 2 is harmful'

        # one temp parameter for keeping track of device

        self.register_buffer(
            '_temp', paddle.to_tensor([0.]), persistable=False)

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

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_unets()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_unets_all()
        return super().load_state_dict(*args, **kwargs)

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
                        lowres_noise_times=None,
                        cond_scale=1.,
                        model_output=None,
                        t_next=None,
                        pred_objective='noise',
                        dynamic_threshold=True):
        assert not (
            cond_scale != 1. and not self.can_classifier_guidance
        ), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = default(model_output, lambda: unet.forward_with_cond_scale(x, noise_scheduler.get_condition(t), text_embeds = text_embeds, text_mask = text_mask, cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_noise_times)))

        if pred_objective == 'noise':
            x_start = noise_scheduler.predict_start_from_noise(
                x, t=t, noise=pred)
        elif pred_objective == 'x_start':
            x_start = pred
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        if dynamic_threshold:
            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
            s = paddle.quantile(
                rearrange(x_start, 'b ... -> b (...)').abs(),
                self.dynamic_thresholding_percentile,
                dim=-1)

            s.clip_(min=1.)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clip(-s, s) / s
        else:
            x_start.clip_(-1., 1.)

        return noise_scheduler.q_posterior(
            x_start=x_start, x_t=x, t=t, t_next=t_next)

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
                 lowres_cond_img=None,
                 lowres_noise_times=None,
                 pred_objective='noise',
                 dynamic_threshold=True):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
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
            lowres_noise_times=lowres_noise_times,
            pred_objective=pred_objective,
            dynamic_threshold=dynamic_threshold)
        noise = paddle.randn(shape=x.shape, dtype=x.dtype)
        # no noise when t == 0
        is_last_sampling_timestep = (t_next == 0) if isinstance(
            noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *(
            (1, ) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance
                                            ).exp() * noise

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
                      cond_scale=1,
                      pred_objective='noise',
                      dynamic_threshold=True):

        batch = shape[0]
        img = paddle.randn(shape)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        timesteps = noise_scheduler.get_sampling_timesteps(batch)

        for times, times_next in tqdm(
                timesteps, desc='sampling loop time step',
                total=len(timesteps)):
            img = self.p_sample(
                unet,
                img,
                times,
                t_next=times_next,
                text_embeds=text_embeds,
                text_mask=text_mask,
                cond_images=cond_images,
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
                lowres_noise_times=lowres_noise_times,
                noise_scheduler=noise_scheduler,
                pred_objective=pred_objective,
                dynamic_threshold=dynamic_threshold)

        img.clip_(-1., 1.)
        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @paddle.no_grad()
    @eval_decorator
    def sample(
            self,
            texts: List[str]=None,
            text_masks=None,
            text_embeds=None,
            cond_images=None,
            batch_size=1,
            cond_scale=1.,
            lowres_sample_noise_level=None,
            stop_at_unet_number=None,
            return_all_unet_outputs=False,
            return_pil_images=False, ):
        self.reset_unets()

        if exists(texts) and not exists(
                text_embeds) and not self.unconditional:
            text_embeds, text_masks = t5_encode_text(
                texts, name=self.text_encoder_name, return_attn_mask=True)

        if not self.unconditional:
            text_masks = default(
                text_masks, lambda: paddle.any(text_embeds != 0., dim=-1))

        if not self.unconditional:
            batch_size = text_embeds.shape[0]

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

        outputs = []

        is_cuda = next(self.parameters()).is_cuda

        lowres_sample_noise_level = default(lowres_sample_noise_level,
                                            self.lowres_sample_noise_level)

        for unet_number, unet, channel, image_size, noise_scheduler, pred_objective, dynamic_threshold in tqdm(
                zip(
                    range(1, len(self.unets) + 1), self.unets, self.
                    sample_channels, self.image_sizes, self.noise_schedulers,
                    self.pred_objectives, self.dynamic_thresholding)):

            context = self.one_unet_in_gpu(
                unet=unet) if is_cuda else nullcontext()

            with context:
                lowres_cond_img = lowres_noise_times = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_noise_times = self.lowres_noise_schedule.get_times(
                        batch_size, lowres_sample_noise_level)

                    lowres_cond_img = resize_image_to(
                        img, image_size, pad_mode='reflect')
                    lowres_cond_img, _ = self.lowres_noise_schedule.q_sample(
                        x_start=lowres_cond_img,
                        t=lowres_noise_times,
                        noise=paddle.randn(
                            shape=lowres_cond_img.shape,
                            dtype=lowres_cond_img.dtype))

                shape = (batch_size, self.channels, image_size, image_size)

                img = self.p_sample_loop(
                    unet,
                    shape,
                    text_embeds=text_embeds,
                    text_mask=text_masks,
                    cond_images=cond_images,
                    cond_scale=cond_scale,
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
                 p2_loss_weight_gamma=0.):
        noise = default(noise, lambda: paddle.randn(shape=x_start.shape, dtype=x_start.dtype))

        # normalize to [-1, 1]

        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t

        x_noisy, log_snr = noise_scheduler.q_sample(
            x_start=x_start, t=times, noise=noise)

        # also noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy, _ = self.lowres_noise_schedule.q_sample(
                x_start=lowres_cond_img,
                t=lowres_aug_times,
                noise=paddle.randn(
                    shape=lowres_cond_img.shape, dtype=lowres_cond_img.dtype))

        # get prediction

        pred = unet.forward(
            x_noisy,
            noise_scheduler.get_condition(times),
            text_embeds=text_embeds,
            text_mask=text_mask,
            cond_images=cond_images,
            lowres_noise_times=self.lowres_noise_schedule.get_condition(
                lowres_aug_times),
            lowres_cond_img=lowres_cond_img_noisy,
            cond_drop_prob=self.cond_drop_prob, )

        # prediction objective

        if pred_objective == 'noise':
            target = noise
        elif pred_objective == 'x_start':
            target = x_start
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        # losses

        losses = self.loss_fn(pred, target, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        # p2 loss reweighting

        if p2_loss_weight_gamma > 0:
            loss_weight = (
                self.p2_loss_weight_k + log_snr.exp())**-p2_loss_weight_gamma
            losses = losses * loss_weight

        return losses.mean()

    def forward(self,
                images,
                texts: List[str]=None,
                text_embeds=None,
                text_masks=None,
                unet_number=None,
                cond_images=None):
        assert images.shape[-1] == images.shape[
            -2], f'the images you pass in must be a square, but received dimensions of {images.shape[2]}, {images.shape[-1]}'
        assert not (
            len(self.unets) > 1 and not exists(unet_number)
        ), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        assert not exists(
            self.only_train_unet_number
        ) or self.only_train_unet_number == unet_number, 'you can only train on unet #{self.only_train_unet_number}'

        unet_index = unet_number - 1

        unet = self.get_unet(unet_number)

        noise_scheduler = self.noise_schedulers[unet_index]
        p2_loss_weight_gamma = self.p2_loss_weight_gamma[unet_index]
        pred_objective = self.pred_objectives[unet_index]
        target_image_size = self.image_sizes[unet_index]
        prev_image_size = self.image_sizes[unet_index -
                                           1] if unet_index > 0 else None
        b, c, h, w = images.shape

        assert images.shape[1] == self.channels
        assert h >= target_image_size and w >= target_image_size

        times = noise_scheduler.sample_random_times(b)

        if exists(texts) and not exists(
                text_embeds) and not self.unconditional:
            assert len(texts) == len(
                images
            ), 'number of text captions does not match up with the number of images given'

            text_embeds, text_masks = t5_encode_text(
                texts, name=self.text_encoder_name, return_attn_mask=True)

        if not self.unconditional:
            text_masks = default(
                text_masks, lambda: paddle.any(text_embeds != 0., dim=-1))

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
            lowres_cond_img = resize_image_to(
                images,
                prev_image_size,
                clamp_range=self.input_image_range,
                pad_mode='reflect')
            lowres_cond_img = resize_image_to(
                lowres_cond_img,
                target_image_size,
                clamp_range=self.input_image_range,
                pad_mode='reflect')

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
            p2_loss_weight_gamma=p2_loss_weight_gamma)


def imagen_base_text2im_64(**kwargs):
    model = Imagen(
        unets=Unet64_300M(),
        image_sizes=(64, ),
        timesteps=1000,
        cond_drop_prob=0.1,
        **kwargs)
    return model


def imagen_text2im_64_2b(**kwargs):
    model = Imagen(
        unets=BaseUnet64(),
        image_sizes=(64, ),
        timesteps=1000,
        cond_drop_prob=0.1,
        **kwargs)
    return model


def imagen_text2im_64_SR256(**kwargs):
    model = Imagen(
        unets=(BaseUnet64(), SRUnet256()),
        image_sizes=(64, 256),
        timesteps=1000,
        cond_drop_prob=0.1,
        **kwargs)
    return model


def imagen_SR256(**kwargs):
    model = Imagen(
        unets=SRUnet256(),
        image_sizes=(256, ),
        timesteps=1000,
        cond_drop_prob=0.1,
        **kwargs)
    return model


def imagen_SR1024(**kwargs):
    model = Imagen(
        unets=SRUnet1024(),
        image_sizes=(1024, ),
        timesteps=1000,
        cond_drop_prob=0.1,
        **kwargs)
    return model
