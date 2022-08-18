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
import os
import random
import time
import sys
import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.engine import Engine

sys.path.append("../../../")
from examples.gpt.tools import parse_args, parse_yaml_auto
from fleetx.optim import lr_scheduler as lr
from fleetx.data.sampler import Stack, Tuple
from fleetx.data.tokenizers import GPTTokenizer
from fleetx.datasets.gpt import create_pretrained_dataset_auto, get_train_data_file
from fleetx.models.gpt_model.modeling_auto import GPTModel, GPTForPretraining, GPTPretrainingCriterion


def generate_model(args):
    # Create the critrion for the gpt model
    model = GPTForPretraining(GPTModel(args))
    criterion = GPTPretrainingCriterion()
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    print('>> total parameters: ', len(model.parameters()))
    return model, criterion, tokenizer


def generate_optimizer(model, args):
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
        apply_decay_param_fun=lambda x: x in decay_params)
    return optimizer


def generate_data_holder(args):
    tokens = paddle.static.InputSpec(
        [args.global_batch_size, args.max_seq_len], "int64", "tokens")
    position_ids = paddle.static.InputSpec(
        [args.global_batch_size, args.max_seq_len], "int64", "position_ids")
    labels = paddle.static.InputSpec(
        [args.global_batch_size, args.max_seq_len], "int64", "labels")
    loss_mask = paddle.static.InputSpec(
        [args.global_batch_size, args.max_seq_len], "float32", "loss_mask")
    return [tokens, position_ids], [labels, loss_mask]


def generate_dist_strategy(args):

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True

    if args.use_recompute:
        dist_strategy.recompute = True

    if args.use_pure_fp16:
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "custom_black_list":
            ["reduce_sum", "c_softmax_with_cross_entropy", "elementwise_div"],
            "init_loss_scaling": args.scale_loss,
            "use_pure_fp16": True,
        }

    if args.use_qat:
        dist_strategy.qat = True
        dist_strategy.qat_configs = {
            'channel_wise_abs_max': True,
            'weight_bits': args.weight_bits,
            'activation_bits': args.activation_bits,
            'not_quant_pattern': ['skip_quant'],
        }

    return dist_strategy


def do_train(args):

    fleet.init(is_collective=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    model, criterion, tokenizer = generate_model(args)
    optimizer = generate_optimizer(model, args)
    inputs, labels = generate_data_holder(args)
    dist_strategy = generate_dist_strategy(args)

    engine = Engine(model, inputs, labels, strategy=dist_strategy)
    engine.prepare(optimizer, criterion)

    for _ in range(args.num_train_epochs):
        files = get_train_data_file(args.input_dir)
        files.sort()
        num_files = len(files)
        for f_id in range(num_files):
            data_file = files[f_id]
            train_data, _, _ = create_pretrained_dataset_auto(
                args, [data_file], tokenizer.eos_token_id)
            engine.fit(train_data,
                       batch_size=args.global_batch_size,
                       collate_fn=Tuple(Stack(), Stack(), Stack(), Stack()),
                       use_cache=True)


if __name__ == "__main__":
    args, _ = parse_yaml_auto(parse_args().config)
    do_train(args)
