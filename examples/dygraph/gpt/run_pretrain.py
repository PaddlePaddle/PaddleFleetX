# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
from paddle.io import DataLoader, Dataset

from package.data import Stack, Tuple, Pad
from package.transformers import GPT2Model, GPT2ForPretraining, GPT2PretrainingCriterion, GPT2ModelPipe
from package.transformers import GPT2Tokenizer, GPT2ChineseTokenizer
from package.utils.log import logger
from data import GPT2Dataset
from args import parse_args
import lr
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

MODEL_CLASSES = {
    "gpt2": (GPT2ForPretraining, GPT2Tokenizer),
    "gpt2-cn": (GPT2ForPretraining, GPT2ChineseTokenizer),
}


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_pretrained_dataset(args, input_path, worker_init, worker_index,
                              worker_num, eod_id):
    train_data = GPT2Dataset(
        file_path=input_path,
        worker_index=worker_index,
        num_samples=args.batch_size * args.max_steps * worker_num,
        eod_id=eod_id,
        max_seq_len=args.max_seq_len,
        seed=args.seed)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_replicas=args.dp_degree,
        rank=worker_index)

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        worker_init_fn=worker_init,
        collate_fn=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()))
    return train_data_loader


def set_seed(args, idx):
    if args.device == "cpu":
        idx = 0
    # else:
    #     idx = paddle.distributed.get_rank()
    random.seed(args.seed + idx)
    np.random.seed(args.seed + idx)
    paddle.seed(args.seed + idx)


def do_train(args):
    paddle.set_device(args.device)
    # if paddle.distributed.get_world_size() > 1:
    #     paddle.distributed.init_parallel_env()

    # worker_index = paddle.distributed.get_rank()
    # worker_num = paddle.distributed.get_world_size()
    # set_seed(args)

    strategy = fleet.DistributedStrategy()

    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.mp_degree,
        "pp_degree": args.pp_degree
    }
    assert args.batch_size % args.micro_batch_size == 0
    strategy.pipeline_configs = {
        "accumulate_steps": args.batch_size // args.micro_batch_size,
        "micro_batch_size": args.micro_batch_size
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    dp_rank = hcg.get_data_parallel_rank()
    worker_index = dp_rank
    worker_num = args.dp_degree

    set_seed(args, dp_rank)
    worker_init = WorkerInitObj(args.seed + dp_rank)
    # worker_init = WorkerInitObj(args.seed + paddle.distributed.get_rank())
    local_seed = args.seed + 1024 + paddle.distributed.get_rank()
    global_seed = args.seed + dp_rank

    tracker = get_rng_state_tracker()
    tracker.add('global_seed', global_seed)
    tracker.add('local_seed', local_seed)

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    eod_id = tokenizer.command_name_map["eod"].Id

    pretrained_models_list = list(
        model_class.pretrained_init_configuration.keys())

    # if args.model_name_or_path in pretrained_models_list:
    #     model = GPT2ForPretraining(
    #         GPT2Model(**model_class.pretrained_init_configuration[
    #             args.model_name_or_path]))
    # else:
    #     model = GPT2ForPretraining.from_pretrained(args.model_name_or_path)

    if args.model_name_or_path in pretrained_models_list:
        config = model_class.pretrained_init_configuration[
            args.model_name_or_path]
        config['num_partitions'] = args.mp_degree
        model = GPT2ForPretraining(GPT2Model(**config))
    else:
        model = model_class.from_pretrained(args.model_name_or_path)

    if args.model_name_or_path in pretrained_models_list:
        config = model_class.pretrained_init_configuration[
            args.model_name_or_path]
        config['num_partitions'] = args.mp_degree
        if args.pp_degree == 1:
            model = GPT2ForPretraining(GPT2Model(**config))
        else:
            config['topology'] = hcg.topology()
            model = GPT2ModelPipe(**config)
    else:
        model = model_class.from_pretrained(args.model_name_or_path)

    # creat the critrion for the gpt model
    criterion = GPT2PretrainingCriterion()

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
        clip = paddle.nn.ClipGradByNorm(clip_norm=args.grad_clip)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in decay_params)

    if paddle.distributed.get_world_size() > 1:
        # model = paddle.DataParallel(model)
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
        # scaler = paddle.amp.GradScaler(init_loss_scaling=2 ** 5)
        scaler = fleet.distributed_scaler(scaler)

    if args.model_name_or_path not in pretrained_models_list:
        opt_dict = paddle.load(
            os.path.join(args.model_name_or_path, "model_state.pdopt"))
        optimizer.set_state_dict(opt_dict)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if (os.path.isfile(os.path.join(args.input_dir, f)) and
                "npz_" not in str(f))
        ]
        files.sort()
        num_files = len(files)
        for f_id in range(num_files):
            data_file = files[f_id]
            train_data_loader = create_pretrained_dataset(
                args,
                data_file,
                worker_init,
                worker_index,
                worker_num,
                eod_id=eod_id)
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                tokens, loss_mask, attention_mask, position_ids, labels = batch

                loss_mask.stop_gradient = True
                attention_mask.stop_gradient = True

                if args.pp_degree == 1:
                    with paddle.amp.auto_cast(
                            args.use_amp,
                            custom_white_list=[
                                "layer_norm", "softmax", "gelu"
                            ]):
                        preds = model(tokens, position_ids, attention_mask)
                        loss = criterion(preds, labels, loss_mask)

                    if args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.minimize(optimizer, loss)
                    else:
                        loss.backward()
                        optimizer.step()

                    lr_scheduler.step()
                    optimizer.clear_grad()
                else:
                    assert not args.use_amp, "Currently, pipeline not support amp"
                    data = [(tokens, position_ids, attention_mask), (labels,
                                                                     loss_mask)]
                    loss = model.train_batch(
                        data, optimizer=optimizer, lr_scheduler=lr_scheduler)

                if global_step % args.logging_steps == 0:
                    per_step_speed = args.logging_steps / (
                        time.time() - tic_train)
                    logger.info(
                        "global step %d, epoch: %d, lr: %.10f, batch: %d, loss: %f, speed: %.2f step/s, ips: %.2f tokens/s"
                        % (global_step, epoch, optimizer.get_lr(), step, loss,
                           per_step_speed,
                           per_step_speed * args.max_seq_len * args.batch_size))
                    tic_train = time.time()

                if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                    if worker_index == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # need better way to get inner model of DataParallel
                        # model_to_save = model._layers if isinstance(
                        #     model, paddle.DataParallel) else model
                        if args.dp_degree * args.mp_degree * args.pp_degree != 1:
                            model_to_save = model._layers
                        else:
                            model_to_save = model
                        logger.info("Save model to %s" % output_dir)
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        paddle.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "model_state.pdopt"))
                if global_step >= args.max_steps:
                    logger.info("The training process is complete.")
                    del train_data_loader
                    return

            del train_data_loader


if __name__ == "__main__":
    args = parse_args(MODEL_CLASSES)
    do_train(args)
