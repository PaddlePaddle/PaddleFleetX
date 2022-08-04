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
from examples.gpt.tools import parse_args, parse_yaml

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.sharding import group_sharded_parallel
from paddle.fluid.dygraph.parallel import sync_params_buffers
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

from fleetx.datasets.gpt import create_pretrained_dataset, get_train_data_file
from fleetx.data.tokenizers import GPTTokenizer
from fleetx.utils import logger
from fleetx.optim import lr_scheduler as lr
from examples.gpt.single.run_pretrain import generate_optimizer, model_optimizer_load, model_forward_backward
from fleetx.models.gpt_model.modeling_3D import GPTModel, GPTForPretraining, GPTPretrainingCriterion, GPTForPretrainingPipe


def set_hyrbid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank):
    assert args.device != "cpu"

    random.seed(basic_seed + data_world_rank)
    np.random.seed(basic_seed + data_world_rank)
    paddle.seed(basic_seed + data_world_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = basic_seed + 123 + mp_rank * 10 + pp_rank * 1000
    global_seed = basic_seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add('global_seed', global_seed)
    tracker.add('local_seed', local_seed)


def generate_model(args):
    if args.pp_degree == 1:
        model = GPTForPretraining(GPTModel(args))
    else:
        hcg = fleet.get_hybrid_communicate_group()
        setattr(args, 'topology', hcg.topology())
        model = GPTForPretrainingPipe(args)

    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    criterion = GPTPretrainingCriterion()
    return model, tokenizer, criterion


def wrap_sharding_2_3(args, model, optimizer, scaler):
    hcg = fleet.get_hybrid_communicate_group()
    dp_group = hcg.get_data_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()

    if args.dp_degree > 1:
        sync_params_buffers(
            model, comm_group=dp_group, src_rank=dp_group.ranks[0])

    level = "p_g_os" if args.sharding_stage == 3 else "os_g"
    return group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level=level,
        scaler=scaler,
        group=sharding_group,
        offload=args.sharding_offload)


def wrap_3D_parallel(model, optimizer, scaler):
    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)
    scaler = fleet.distributed_scaler(scaler) if scaler is not None else scaler
    return model, optimizer, scaler


def optim_update_params(args, model, optimizer, scaler):
    if args.sharding_stage in [2, 3] and args.dp_degree > 1:
        hcg = fleet.get_hybrid_communicate_group()
        dp_group = hcg.get_data_parallel_group()
        fused_allreduce_gradients(model.parameters(), hcg)
        if args.sharding_stage == 3:
            for p in model.parameters():
                if hasattr(p, "bw_storage"):
                    assert p.grad is None, "This case shouldn't happen."
                    p.bw_storage.scale_(1.0 / dp_group.nranks)
                    paddle.distributed.all_reduce(p.bw_storage, group=dp_group)

    if args.use_pure_fp16:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def model_optimizer_save(args, model, optimizer, tokenizer, global_step):
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    pp_rank = hcg.get_stage_id()
    sharding_rank = hcg.get_sharding_parallel_rank()

    # TODO: 1. merge paramters while saving model. 2. ensure that the model is saved and loaded correctly
    # only dp_rank = 0 save model
    model_to_save = model._layers if paddle.distributed.get_world_size(
    ) > 1 and args.sharding_stage not in [2, 3] else model
    output_dir = os.path.join(args.output_dir, "step_%d" % global_step)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Save model to %s" % output_dir)

    if args.pp_degree > 1:
        paddle.save(model.state_dict(),
                    os.path.join(
                        output_dir,
                        "model_mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}.pdparams"
                        .format(mp_rank, sharding_rank, pp_rank)))
        paddle.save(
            optimizer.state_dict(),
            os.path.join(
                output_dir,
                "model_state_mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}.pdopt"
                .format(mp_rank, sharding_rank, pp_rank)))
    else:
        if args.sharding_stage == 3:
            # If parameter need to convert to cpu, please add convert2cpu=True
            model_to_save.get_all_parameters(convert2cpu=False)
        paddle.save(model.state_dict(),
                    os.path.join(output_dir,
                                 "model_mp_{:0>2d}_sharding_{:0>2d}.pdparams"
                                 .format(mp_rank, sharding_rank)))
        paddle.save(
            optimizer.state_dict(),
            os.path.join(output_dir,
                         "model_state_mp_{:0>2d}_sharding_{:0>2d}.pdopt"
                         .format(mp_rank, sharding_rank)))


@paddle.no_grad()
def run_evaluate(args,
                 data_loader,
                 model,
                 criterion,
                 iter_steps,
                 global_step,
                 epoch,
                 task_name="valid"):
    model.eval()
    all_loss = []
    local_time = time.time()
    for eval_step, batch in enumerate(data_loader):
        tokens, loss_mask, position_ids, labels = batch
        if args.pp_degree < 2:
            preds = model(tokens, position_ids)
            loss = criterion(preds, labels, loss_mask)
        else:
            data = [(tokens, position_ids), (labels, loss_mask)]
            loss = model.eval_batch(data, compute_loss=True)

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

    nranks = paddle.distributed.get_world_size()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.mp_degree,
        "pp_degree": args.pp_degree,
        "sharding_degree": args.sharding_degree,
    }

    accumulate_steps = args.local_batch_size // args.micro_batch_size
    strategy.pipeline_configs = {
        "accumulate_steps": accumulate_steps,
        "micro_batch_size": args.micro_batch_size
    }

    # set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}
    fleet.init(is_collective=True, strategy=strategy)

    # obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    pp_rank = hcg.get_stage_id()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()
    sharding_size = hcg.get_sharding_parallel_world_size()

    data_world_rank = dp_rank * sharding_size + sharding_rank
    data_world_size = args.dp_degree * args.sharding_degree
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, data_world_rank, mp_rank, pp_rank)
    default_global_tokens_num = args.global_batch_size * args.max_seq_len

    model, tokenizer, criterion = generate_model(args)
    optimizer, lr_scheduler = generate_optimizer(model, args)

    if args.use_pure_fp16:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
        model = paddle.amp.decorate(
            models=model, level='O2', save_dtype='float32')
    else:
        scaler = None

    # wrap sharding stage2/3 and add collective group
    if args.sharding_stage in [2, 3]:
        assert args.mp_degree == args.pp_degree == 1, "sharding stage2/3 will support hybrid parallel later"
        model, optimizer, scaler = wrap_sharding_2_3(args, model, optimizer,
                                                     scaler)
    else:
        model, optimizer, scaler = wrap_3D_parallel(model, optimizer, scaler)

    model, optimizer = model_optimizer_load(args, model, optimizer)

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
                data_world_size=data_world_size,
                data_world_rank=data_world_rank,
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

                if args.pp_degree == 1:
                    if args.use_recompute and isinstance(model,
                                                         paddle.DataParallel):
                        with model.no_sync():
                            loss = model_forward_backward(
                                args, model, criterion, tokens, position_ids,
                                labels, loss_mask, scaler)
                        fused_allreduce_gradients(
                            list(model.parameters()), None)
                    else:
                        loss = model_forward_backward(
                            args, model, criterion, tokens, position_ids,
                            labels, loss_mask, scaler)
                    optim_update_params(args, model, optimizer, scaler)

                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.clear_grad()
                else:
                    data = [(tokens, position_ids), (labels, loss_mask)]
                    with paddle.amp.auto_cast(
                            args.use_pure_fp16,
                            custom_black_list=[
                                "reduce_sum", "c_softmax_with_cross_entropy",
                                "elementwise_div"
                            ],
                            level='O2'):
                        loss = model.train_batch(
                            data,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            scaler=scaler if args.use_pure_fp16 else None)

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
                        %
                        (global_step, epoch, step, avg_loss, avg_reader_cost,
                         1. / speed, speed, speed * default_global_tokens_num,
                         speed * default_global_tokens_num / nranks,
                         optimizer.get_lr()))

                    train_reader_cost = 0.0
                    train_run_cost = 0.0

                if global_step % args.eval_freq == 0:
                    # Since the valid data broardcast to all devices, we do evaluate on all device.
                    run_evaluate(args, valid_data_loader, model, criterion,
                                 args.eval_iters, global_step, epoch, "valid")

                # TODO: 1. merge paramters while saving model. 2. ensure that the model is saved and loaded correctly
                # only dp_rank = 0 save model
                if (global_step % args.save_steps == 0 or
                        global_step >= args.max_steps) and dp_rank == 0:
                    model_optimizer_save(args, model, optimizer, tokenizer,
                                         global_step)

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
