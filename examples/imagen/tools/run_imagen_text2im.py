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

import argparse
import datetime
import numpy as np
import math
import time
import json
import os
import sys
import random
from typing import Iterable, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

import packages.utils as utils
import packages.misc as misc

from packages.model_ema import ModelEma
from packages.optim_factory import create_optimizer

sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

from fleetx.models.imagen_model import modeling_imagen_text2im
from fleetx.datasets.imagen import collate_imagen_base64, build_imagen_train_dataset


def get_args():
    parser = argparse.ArgumentParser(
        'Imagen text2image 64x64 training script', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=9, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)

    parser.add_argument('--use_pure_fp16', action='store_true', default=False)
    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument(
        '--enable_multi_print',
        action='store_true',
        default=False,
        help='allow each gpu prints something')
    parser.add_argument('--dp_degree', default=1, type=int)
    parser.add_argument('--mp_degree', default=1, type=int)
    parser.add_argument('--pp_degree', default=1, type=int)
    parser.add_argument('--sharding_stage', default=0, type=int)
    parser.add_argument('--sharding_degree', default=8, type=int)
    parser.add_argument(
        '--sharding_offload', action='store_true', default=False)

    # Model parameters
    parser.add_argument("--model", type=str, default="imagen_base_text2im_64")
    parser.add_argument(
        '--unet_number',
        default=1,
        type=int,
        metavar='UNetNumber',
        help='Unet Number (default: 1)')
    parser.add_argument('--text_encoder_name', default='t5/t5-11b', type=str)
    parser.add_argument('--text_embed_dim', default=1024, type=int)
    parser.add_argument(
        '--channels',
        default=3,
        type=int,
        metavar='CHANNELS',
        help='input channels (default: 3)')
    parser.add_argument(
        '--timesteps',
        type=int,
        default=1000,
        metavar='TIMESTEPS',
        help='Timesteps (default: 1000)')
    parser.add_argument(
        '--cond_drop_prob',
        type=float,
        default=0.1,
        metavar='CDP',
        help='Condition Drop Probility(default: 0.1)')
    parser.add_argument(
        '--loss_type', type=str, default='l2', help='[l1, l2, huber]')
    parser.add_argument('--noise_schedules', type=str, default='cosine')
    parser.add_argument('--pred_objectives', type=str, default='noise')
    parser.add_argument('--lowres_noise_schedule', type=str, default='linear')
    parser.add_argument('--lowres_sample_noise_level', type=float, default=0.2)
    parser.add_argument(
        '--per_sample_random_aug_noise_level',
        action='store_true',
        default=False)
    parser.add_argument(
        '--condition_on_text', action='store_true', default=False)
    parser.add_argument(
        '--auto_normalize_img', action='store_true', default=False)
    parser.add_argument(
        '--continuous_times', action='store_true', default=False)
    parser.add_argument('--p2_loss_weight_gamma', type=float, default=0.5)
    parser.add_argument('--p2_loss_weight_k', type=float, default=1.0)
    parser.add_argument(
        '--dynamic_thresholding', action='store_true', default=False)
    parser.add_argument(
        '--dynamic_thresholding_percentile', type=float, default=0.9)
    parser.add_argument('--only_train_unet_number', default=None)
    parser.add_argument(
        '--input_resolution',
        default=64,
        type=int,
        metavar='INPUT_RESOLUTION',
        help='images input size (default: 64)')
    parser.add_argument(
        '--second_resolution',
        default=256,
        type=int,
        metavar='SECOND_RESOLUTION',
        help='Super Resolution (default: 256)')
    parser.add_argument(
        '--super_resolution', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument(
        '--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument(
        '--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument(
        '--opt',
        default='adam',
        type=str,
        nargs='+',
        metavar='OPTIMIZER',
        help='Optimizers (default: adam)')
    parser.add_argument(
        '--opt_eps',
        default=1e-8,
        type=float,
        metavar='EPSILON',
        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument(
        '--weight_decay',
        default=0,
        type=float,
        metavar='WD',
        help='Weight decay(default: 0)')
    parser.add_argument(
        '--opt_betas',
        default=None,
        type=float,
        nargs='+',
        metavar='BETA',
        help='Optimizer Betas (default: 0.9, 0.99, use opt default)')
    parser.add_argument(
        '--clip_grad',
        type=float,
        default=None,
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument(
        '--group_wd_params',
        action='store_true',
        default=False,
        help='group wd parameters')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        metavar='LR',
        help='learning rate (default: 1e-4)')
    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='warmup learning rate (default: 1e-6)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=0,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=5,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=-1,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument(
        '--train_interpolation',
        type=str,
        default='bicubic',
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")'
    )

    # Dataset parameters
    parser.add_argument(
        '--data_path', default='FOOD101', type=str, help='dataset path')
    parser.add_argument(
        '--input_format', default='files', type=str, help='input format')
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument("--text_max_len", type=int, default=128)
    parser.add_argument('--filter_image_resolution', default=128, type=int)
    parser.add_argument(
        '--output_dir',
        default='',
        help='path where to save, empty for no saving')
    parser.add_argument(
        '--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument(
        '--device', default='gpu', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument(
        '--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument(
        '--exp_name',
        default='imagen_base',
        type=str,
        help='name of exp. it is helpful when save the checkpoint')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = modeling_imagen_text2im.__dict__[args.model](
        text_encoder_name=args.text_encoder_name,
        in_chans=args.channels,
        loss_type=args.loss_type,
        noise_schedules=args.noise_schedules,
        pred_objectives=args.pred_objectives,
        lowres_noise_schedule=args.lowres_noise_schedule,
        lowres_sample_noise_level=args.lowres_sample_noise_level,
        per_sample_random_aug_noise_level=args.
        per_sample_random_aug_noise_level,
        condition_on_text=args.condition_on_text,
        auto_normalize_img=args.auto_normalize_img,
        continuous_times=args.continuous_times,
        p2_loss_weight_gamma=args.p2_loss_weight_gamma,
        p2_loss_weight_k=args.p2_loss_weight_k,
        dynamic_thresholding=args.dynamic_thresholding,
        dynamic_thresholding_percentile=args.dynamic_thresholding_percentile,
        only_train_unet_number=args.only_train_unet_number, )

    return model


def main(args):
    # sharding or distributed init
    if args.sharding_stage in [2, 3]:
        utils.init_sharding(args)
    else:
        utils.init_distributed_mode(args)

    print("{}".format(args).replace(', ', ',\n'))

    device = paddle.set_device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # get_model    
    model = get_model(args)

    # get dataset
    dataset_train = build_imagen_train_dataset(args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        num_training_steps_per_epoch = len(
            dataset_train) // args.batch_size // num_tasks

        if len(dataset_train) % num_tasks != 0:
            print(
                'Warning: Enabling distributed inference with an infer dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
    else:
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = misc.VisualdlLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = paddle.io.DataLoader(
        dataset_train,  #batch_sampler=sampler_train,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=True,
        num_workers=args.num_workers,
        use_shared_memory=True,
        collate_fn=collate_imagen
        if args.input_format == "files" else collate_imagen_base64,
        places=device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model.unet,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel().item() for p in model.parameters()
                       if not p.stop_gradient)

    MB = 1024.0 * 1024.0
    print("Model = %s" % str(model_without_ddp))
    print('Number of params: %f MB' % (n_parameters / MB))

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size(
    )

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" %
          num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" %
          (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model.unets)

    if args.use_pure_fp16:
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=2.**16,
            incr_every_n_steps=2000,
            decr_every_n_nan_or_inf=1, )
        model = paddle.amp.decorate(
            models=model, level='O2', save_dtype='float32')

    scaler = scaler if args.use_pure_fp16 else None
    if args.sharding_stage in [2, 3]:
        model, optimizer, scaler = utils.wrap_sharding_2_3(
            model, optimizer, scaler, args.sharding_stage,
            args.sharding_offload)
    else:
        if args.distributed:
            model = paddle.DataParallel(model)
            model_without_ddp = model._layers

    print("Use step level LR & WD scheduler!")

    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps, )

    utils.auto_load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        scaler=scaler,
        model_ema=model_ema)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch *
                                args.update_freq)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            epoch,
            scaler,
            args.clip_grad,
            model_ema,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            args=args, )
        if args.output_dir:
            if (epoch + 1
                ) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=scaler,
                    epoch=epoch,
                    exp_name=args.exp_name)

        log_stats = {
            **
            {f'train_{k}': v
             for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                    os.path.join(args.output_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    exit(0)


def train_one_epoch(model: paddle.nn.Layer,
                    data_loader: Iterable,
                    optimizer,
                    epoch: int,
                    loss_scaler,
                    max_norm: float=0,
                    model_ema: Optional[ModelEma]=None,
                    log_writer=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    num_training_steps_per_epoch=None,
                    update_freq=None,
                    args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    optimizer.clear_grad()

    for data_iter_step, batch in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            optimizer.set_lr(lr_schedule_values[it])

        if args.super_resolution:
            sr_samples, samples, text_embeds, text_masks = batch
        else:
            samples, text_embeds, text_masks = batch

        with paddle.amp.auto_cast(
                args.use_pure_fp16,
                custom_black_list=[
                    "reduce_sum", "c_softmax_with_cross_entropy",
                    "elementwise_div"
                ],
                custom_white_list=["fused_attention", "fused_feedforward"],
                level='O2'):
            loss = model(
                images=samples,
                text_embeds=text_embeds,
                text_masks=text_masks,
                unet_number=args.unet_number)

        loss /= update_freq
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler:
            loss_scaler.scale(loss).backward()
            if args.sharding_stage in [2, 3]:
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                loss_scaler.minimize(optimizer, loss)
            grad_norm = None
            loss_scale_value = loss_scaler.state_dict()["scale"].item()
        else:
            loss.backward()
            optimizer.step()
            if (data_iter_step + 1) % update_freq == 0:
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = None

        optimizer.clear_grad()
        if (data_iter_step + 1) % update_freq == 0:
            if model_ema is not None:
                model_ema.update(model)

        paddle.device.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if loss_scaler is not None:
            metric_logger.update(loss_scale=loss_scale_value)
        lr = optimizer.get_lr()
        metric_logger.update(lr=lr)
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=lr, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(now_time, "Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    opts = get_args()
    main(opts)
