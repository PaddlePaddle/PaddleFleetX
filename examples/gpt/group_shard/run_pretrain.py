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
from tools import parse_args, parse_yaml

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.sharding import group_sharded_parallel

from fleetx.datasets.gpt import create_pretrained_dataset, get_train_data_file
from fleetx.utils import logger
from fleetx.optim import lr_scheduler as lr
from examples.gpt.single.run_pretrain import run_evaluate, generate_model, generate_optimizer
from examples.gpt.single.run_pretrain import model_optimizer_load, model_forward_backward


def set_group_sharded_parallel_seed(basic_seed, sharding_rank):
    assert args.device != "cpu"

    random.seed(basic_seed + sharding_rank)
    np.random.seed(basic_seed + sharding_rank)
    paddle.seed(basic_seed + sharding_rank)


def wrap_sharding_2_3(model, optimizer, scaler, sharding_offload):
    group = fleet.get_hybrid_communicate_group().get_sharding_parallel_group()
    level = "p_g_os" if args.sharding_stage == 3 else "os_g"
    return group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level=level,
        scaler=scaler,
        group=group,
        offload=sharding_offload)


def group_sharded_model_optimizer_save(args, model, optimizer, global_step,
                                       sharding_rank):
    model_to_save = model._layers if paddle.distributed.get_world_size(
    ) > 1 and args.sharding_stage not in [2, 3] else model
    output_dir = os.path.join(args.output_dir, "step_%d" % global_step)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Save model to %s" % output_dir)

    if args.sharding_stage == 3:
        # If parameter need to convert to cpu, please add convert2cpu=True
        model_to_save.get_all_parameters(convert2cpu=False)
    if sharding_rank == 0:
        tokenizer.save_pretrained(output_dir)
    model_to_save.save_pretrained(output_dir)
    paddle.save(
        optimizer.state_dict(),
        os.path.join(
            output_dir,
            "model_state_sharding_{:0>2d}.pdopt".format(sharding_rank)))


def do_train(args):
    paddle.set_device(args.device)
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.mp_degree,
        "pp_degree": args.pp_degree,
        "sharding_degree": args.sharding_degree
    }
    default_global_tokens_num = args.global_batch_size * args.max_seq_len

    # sharding stage2/3 not support hybrid parallel
    if args.sharding_stage in [2, 3]:
        assert args.dp_degree == args.mp_degree == args.pp_degree == 1, "sharding stage2/3 will support hybrid parallel later"

    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    sharding_rank = hcg.get_sharding_parallel_rank()
    set_group_sharded_parallel_seed(args.seed, sharding_rank)
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    model, criterion, tokenizer = generate_model(args)
    optimizer, lr_scheduler = generate_optimizer(model, args)
    model, optimizer = model_optimizer_load(model, optimizer, args)

    if args.use_pure_fp16:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
        model = paddle.amp.decorate(
            models=model, level='O2', save_dtype='float32')
    else:
        scaler = None

    # wrap sharding stage2/3 and add collective group
    if args.sharding_stage in [2, 3]:
        scaler = scaler if args.use_pure_fp16 else None
        model, optimizer, scaler = wrap_sharding_2_3(model, optimizer, scaler,
                                                     args.sharding_offload)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        files = get_train_data_file(args)
        files.sort()
        num_files = len(files)

        for f_id in range(num_files):
            data_file = files[f_id]
            train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
                args, [data_file],
                local_rank=local_rank,
                data_world_size=args.sharding_degree,
                data_world_rank=sharding_rank,
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

                loss = model_forward_backward(args, model, criterion, tokens,
                                              position_ids, labels, loss_mask,
                                              scaler)

                if args.use_pure_fp16:
                    if args.sharding_stage in [2, 3]:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
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

                if (global_step % args.save_steps == 0 or
                        global_step >= args.max_steps):
                    group_sharded_model_optimizer_save(
                        args, model, optimizer, global_step, sharding_rank)

                if global_step >= args.max_steps:
                    run_evaluate(args, test_data_loader, model, criterion,
                                 args.test_iters, global_step, epoch, "test")
                    logger.info("The training process is complete.")
                    del train_data_loader
                    return
                reader_start = time.time()
            del train_data_loader


if __name__ == "__main__":
    args = parse_yaml(parse_args().config)
    do_train(args)
