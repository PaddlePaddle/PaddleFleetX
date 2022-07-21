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
import math
import os
import random
import time
import sys
sys.path.append("..")
from args import parse_args

import numpy as np
import paddle
from fleetx.models.gpt_model import GPTModel, GPTForPretraining, GPTPretrainingCriterion
from fleetx.datasets.gpt import create_pretrained_dataset, get_train_data_file
from fleetx.data.tokenizers import GPTTokenizer
from fleetx.utils import logger
from fleetx.optim import lr_scheduler as lr


@paddle.no_grad()
def run_evaluate(args,
                 data_loader,
                 model,
                 criterion,
                 iter_steps,
                 global_step,
                 epoch,
                 task_name="valid"):
    """
    evaluate for gpt model
    """
    model.eval()
    all_loss = []
    local_time = time.time()
    for eval_step, batch in enumerate(data_loader):
        tokens, loss_mask, position_ids, labels = batch
        preds = model(tokens, position_ids)
        preds = paddle.cast(preds, dtype="float32")
        loss = criterion(preds, labels, loss_mask)
        all_loss.append(float(loss))
        if eval_step >= iter_steps - 1:
            break
    average_loss = sum(all_loss) / len(all_loss)
    logger.info(
        "%s step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s" %
        (task_name, global_step, epoch, eval_step, average_loss,
         iter_steps / (time.time() - local_time)))
    model.train()


def do_train(args):
    paddle.set_device(args.device)
    accumulate_steps = args.local_batch_size // args.micro_batch_size
    default_global_tokens_num = args.global_batch_size * args.max_seq_len

    # Create the critrion for the gpt model
    model = GPTForPretraining(GPTModel(args))
    criterion = GPTPretrainingCriterion()
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    print('>> total parameters: ', len(model.parameters()))

    if args.decay_steps is None:
        args.decay_steps = args.max_steps
    warmup_step = args.warmup_rate * args.decay_steps
    lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_step=warmup_step,
        decay_step=args.decay_steps)
    clip = None
    if args.grad_clip > 0:
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.grad_clip)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler
        if lr_scheduler is not None else args.max_lr,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in decay_params,
        # TODO: remove 'multi_precision' in definition of optimizer
        # and add it to 'paddle.amp.decorate'
        multi_precision=args.use_pure_fp16)

    if args.ckpt_dir:
        logger.info("Try to load checkpoint from %s " % args.ckpt_dir)
        model_path = os.path.join(args.ckpt_dir, "model.pdparams")
        opt_path = os.path.join(args.ckpt_dir, "model_state.pdopt")
        if os.path.exists(model_path):
            model_dict = paddle.load(model_path)
            model.set_state_dict(model_dict)
        else:
            logger.warning("No optimizer checkpoint file found in %s." %
                           model_path)

        if os.path.exists(opt_path):
            opt_dict = paddle.load(opt_path)
            optimizer.set_state_dict(opt_dict)
        else:
            logger.warning("No optimizer checkpoint file found in %s." %
                           opt_path)

    if args.use_pure_fp16:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
        model = paddle.amp.decorate(
            models=model, level='O2', save_dtype='float32')

    global_step = 0
    for epoch in range(args.num_train_epochs):
        files = get_train_data_file(args)
        files.sort()
        num_files = len(files)

        for f_id in range(num_files):
            data_file = files[f_id]
            train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
                args, [data_file],
                local_rank=0,
                data_world_size=1,
                data_world_rank=0,
                max_seq_len=args.max_seq_len,
                eos_id=tokenizer.eos_token_id)
            # Bug fix, if not call valid_data_loader, the enumerate will call valid_data_loader
            # many times. and start a new random dataloader.
            valid_data_loader = valid_data_loader()
            test_data_loader = test_data_loader()

            # time count
            train_reader_cost = 0.0
            train_run_cost = 0.0
            reader_start = time.time()
            for step, batch in enumerate(train_data_loader()):
                train_reader_cost += time.time() - reader_start
                train_start = time.time()

                global_step += 1
                tokens, loss_mask, position_ids, labels = batch

                loss_mask.stop_gradient = True
                labels.stop_gradient = True
                position_ids.stop_gradient = True
                loss = 0.0
                for i in range(accumulate_steps):
                    start_index = i * args.micro_batch_size
                    end_index = start_index + args.micro_batch_size
                    with paddle.amp.auto_cast(
                            args.use_pure_fp16,
                            custom_black_list=[
                                "reduce_sum", "c_softmax_with_cross_entropy",
                                "elementwise_div"
                            ],
                            level='O2'):
                        preds = model(tokens[start_index:end_index, :],
                                      position_ids[start_index:end_index, :])
                        loss_mbs = criterion(
                            preds, labels[start_index:end_index, :],
                            loss_mask[start_index:end_index, :])
                    loss_mbs = loss_mbs / accumulate_steps
                    if args.use_pure_fp16:
                        scaler.scale(loss_mbs).backward()
                    else:
                        loss_mbs.backward()
                    loss = loss + loss_mbs

                if args.use_pure_fp16:
                    scaler.minimize(optimizer, loss)
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                # Sync for profile time, delete it may be a little faster
                paddle.device.cuda.synchronize()
                train_run_cost += time.time() - train_start

                if global_step % args.logging_freq == 0:
                    avg_loss = loss.numpy()
                    speed = args.logging_freq / (
                        train_reader_cost + train_run_cost)
                    avg_reader_cost = train_reader_cost / args.logging_freq

                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %.9f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
                        % (global_step, epoch, step, avg_loss, avg_reader_cost,
                           1. / speed, speed,
                           speed * default_global_tokens_num, speed *
                           default_global_tokens_num, optimizer.get_lr()))
                    # tic_train = time.time()
                    train_reader_cost = 0.0
                    train_run_cost = 0.0

                if global_step % args.eval_freq == 0:
                    # Since the valid data broardcast to all devices, we do evaluate on all device.
                    run_evaluate(args, valid_data_loader, model, criterion,
                                 args.eval_iters, global_step, epoch, "valid")

                # model save
                if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                    output_dir = os.path.join(args.output_dir,
                                              "model_%d" % global_step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info("Save model to %s" % output_dir)
                    paddle.save(model.state_dict(),
                                os.path.join(output_dir, "model.pdparams"))
                    paddle.save(optimizer.state_dict(),
                                os.path.join(output_dir, "model_state.pdopt"))

                if global_step >= args.max_steps:
                    run_evaluate(args, test_data_loader, model, criterion,
                                 args.test_iters, global_step, epoch, "test")
                    logger.info("The training process is complete.")
                    del train_data_loader
                    return

                reader_start = time.time()

            del train_data_loader


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
