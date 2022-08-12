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
import yaml
import numpy as np

import paddle
sys.path.append("../../../")
from examples.gpt.tools import parse_args, parse_yaml
from fleetx.models.gpt_model.modeling import GPTModel, GPTForPretraining, GPTPretrainingCriterion
from fleetx.datasets.gpt import create_pretrained_dataset, get_train_data_file
from fleetx.data.tokenizers import GPTTokenizer
from fleetx.utils import logger
from fleetx.optim import lr_scheduler as lr
from fleetx.core import FleetxModule, Engine


class GPTModule(FleetxModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = GPTForPretraining(GPTModel(args))
        self.loss_fn = GPTPretrainingCriterion()

        print('>> total parameters: ', len(self.model.parameters()))

    def forward(self, tokens, ids):
        return self.model(tokens, ids)

    def training_step(self, batch):
        tokens, loss_mask, position_ids, labels = batch

        loss_mask.stop_gradient = True
        labels.stop_gradient = True
        position_ids.stop_gradient = True

        preds = self(tokens, position_ids)
        loss = self.loss_fn(preds, labels, loss_mask)

        return loss

    def training_step_end(self, loss, global_step, epoch, step, reader_cost,
                          run_cost):
        avg_loss = loss.numpy()
        speed = self.args.logging_freq / (reader_cost + run_cost)
        avg_reader_cost = reader_cost / self.args.logging_freq
        default_global_tokens_num = self.args.global_batch_size * self.args.max_seq_len

        logger.info(
            "global step %d, epoch: %d, batch: %d, loss: %.9f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
            % (global_step, epoch, step, avg_loss, avg_reader_cost, 1. / speed,
               speed, speed * default_global_tokens_num,
               speed * default_global_tokens_num, self.optimizer.get_lr()))

    def configure_optimizers(self):
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
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler
            if lr_scheduler is not None else args.max_lr,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            parameters=self.model.parameters(),
            weight_decay=args.weight_decay,
            grad_clip=clip,
            apply_decay_param_fun=lambda x: x in decay_params,
            multi_precision=args.use_pure_fp16)
        return optimizer, lr_scheduler


def do_train(args):
    paddle.set_device(args.device)

    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    global_config = yaml.load(
        open(parse_args().config, 'rb'), Loader=yaml.Loader)

    module = GPTModule(args)
    engine = Engine(module=module, configs=global_config['PreTraining'])

    if args.ckpt_dir:
        engine.load(ckpt_dir=args.ckpt_dir)

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

            engine.fit(train_data_loader=train_data_loader,
                       epoch=epoch,
                       global_step=global_step)

            del train_data_loader


if __name__ == "__main__":
    args = parse_yaml(parse_args().config)
    do_train(args)
