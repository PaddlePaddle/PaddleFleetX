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
import copy

import paddle
from paddle.distributed import fleet

sys.path.append("../../../")
from fleetx.utils import logger
from fleetx.optim import lr_scheduler as lr
from fleetx.core.module.basic_module import BasicModule
from fleetx.utils.tensor_fusion_helper import fused_parameters
from fleetx.models.imagen_model import modeling


class ImagenModule(BasicModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        model_config = copy.deepcopy(configs['Model'])
        model_name = model_config.pop('name')

        self.model = getattr(modeling, model_name)(**model_config)
        self.loss_fn = self.model.loss_fn

        print('>> total parameters: ', len(self.model.parameters()))

    def forward(self, samples, text_embeds, text_masks):
        return self.model(
            samples, text_embeds=text_embeds, text_masks=text_masks)

    def training_step(self, batch):
        samples, text_embeds, text_masks = batch
        return self(samples, text_embeds, text_masks)

    def training_step_end(self, log_dict):
        speed = self.configs['Engine']['logging_freq'] / log_dict['train_cost']

        logger.info(
            "[train] global step %d, epoch: %d, batch: %d, loss: %.9f, avg_batch_cost: %.5f sec, speed: %.2f step/s, learning rate: %.5e"
            % (self.global_step, log_dict['epoch'], log_dict['batch'],
               log_dict['loss'], 1. / speed, speed, self.optimizer.get_lr()))

    def configure_optimizers(self):
        self.decay_fused_tensors, self.all_fused_tensors = None, None

        if self.configs['Fused']['tensor_fusion']:
            self.decay_fused_tensors, self.all_fused_tensors = fused_parameters(
                self.model)

        opt_configs = self.configs['Optimizer']
        warmup_step = opt_configs['lr']['warmup_steps']
        lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
            max_lr=opt_configs['lr']['max_lr'],
            min_lr=opt_configs['lr']['min_lr'],
            warmup_step=warmup_step,
            decay_step=opt_configs['lr']['decay_steps'])

        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=opt_configs[
            'grad_clip']) if opt_configs['grad_clip'] > 0 else None

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        if self.configs['Fused']['tensor_fusion']:
            decay_params = [p.name for p in self.decay_fused_tensors]
        else:
            decay_params = [
                p.name for n, p in self.model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ]

        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler
            if lr_scheduler is not None else opt_configs['lr']['max_lr'],
            beta1=opt_configs['adam_beta1'],
            beta2=opt_configs['adam_beta2'],
            epsilon=opt_configs['adam_epsilon'],
            parameters=self.all_fused_tensors
            if self.configs['Fused']['tensor_fusion'] else
            self.model.parameters(),
            weight_decay=opt_configs['weight_decay'],
            grad_clip=clip,
            apply_decay_param_fun=lambda x: x in decay_params,
            multi_precision=self.configs['Engine']['mix_precision'][
                'use_pure_fp16'])
        return optimizer, lr_scheduler

    def validation_step(self, batch):
        tokens, position_ids, labels, loss_mask = batch
        preds = self(tokens, position_ids)
        preds = paddle.cast(preds, dtype="float32")
        loss = self.loss_fn(preds, labels, loss_mask)
        return loss

    def validation_step_end(self, log_dict):
        speed = self.configs['Engine']['logging_freq'] / log_dict['eval_cost']
        logger.info(
            "[eval] epoch: %d, batch: %d, loss: %.9f, avg_eval_cost: %.5f sec, speed: %.2f step/s"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'],
               1. / speed, speed))

    def test_step(self, batch):
        tokens, position_ids, labels, loss_mask = batch
        preds = self(tokens, position_ids)
        preds = paddle.cast(preds, dtype="float32")
        loss = self.loss_fn(preds, labels, loss_mask)
        return loss

    def test_step_end(self, log_dict):
        speed = self.configs['Engine']['logging_freq'] / log_dict['test_cost']
        logger.info(
            "[test] epoch: %d, batch: %d, loss: %.9f, avg_test_cost: %.5f sec, speed: %.2f step/s"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'],
               1. / speed, speed))
