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

import sys

import paddle
sys.path.append("../../../")
from fleetx.models.gpt_model.modeling import GPTModel, GPTForPretraining, GPTPretrainingCriterion
from fleetx.utils import logger
from fleetx.optim import lr_scheduler as lr
from fleetx.core.module.basic_module import BasicModule


class GPTModule(BasicModule):
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

    def training_step_end(self, loss, epoch, step, reader_cost, train_cost):
        avg_loss = loss.numpy()
        speed = self.args.logging_freq / (reader_cost + train_cost)
        avg_reader_cost = reader_cost / self.args.logging_freq
        default_global_tokens_num = self.args.global_batch_size * self.args.max_seq_len

        logger.info(
            "[train] global step %d, epoch: %d, batch: %d, loss: %.9f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
            % (self.global_step, epoch, step, avg_loss, avg_reader_cost,
               1. / speed, speed, speed * default_global_tokens_num,
               speed * default_global_tokens_num, self.optimizer.get_lr()))

    def configure_optimizers(self):
        if self.args.decay_steps is None:
            self.args.decay_steps = self.args.max_steps
        warmup_step = self.args.warmup_rate * self.args.decay_steps
        lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
            max_lr=self.args.max_lr,
            min_lr=self.args.min_lr,
            warmup_step=warmup_step,
            decay_step=self.args.decay_steps)
        clip = None
        if self.args.grad_clip > 0:
            clip = paddle.nn.ClipGradByGlobalNorm(
                clip_norm=self.args.grad_clip)

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler
            if lr_scheduler is not None else self.args.max_lr,
            beta1=self.args.adam_beta1,
            beta2=self.args.adam_beta2,
            epsilon=self.args.adam_epsilon,
            parameters=self.model.parameters(),
            weight_decay=self.args.weight_decay,
            grad_clip=clip,
            apply_decay_param_fun=lambda x: x in decay_params,
            multi_precision=self.args.use_pure_fp16)
        return optimizer, lr_scheduler

    def validation_step(self, batch):
        tokens, loss_mask, position_ids, labels = batch
        preds = self.model(tokens, position_ids)
        preds = paddle.cast(preds, dtype="float32")
        loss = self.loss_fn(preds, labels, loss_mask)
        return loss

    def validation_step_end(self, loss, epoch, step, eval_cost):
        speed = self.args.logging_freq / eval_cost
        logger.info(
            "[eval] step %d, epoch: %d, batch: %d, loss: %.9f, avg_eval_cost: %.5f sec, speed: %.2f step/s"
            % (self.global_step, epoch, step, loss, 1. / speed, speed))
