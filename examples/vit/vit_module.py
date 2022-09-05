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
from paddle import nn
from paddle.distributed import fleet

from fleetx.utils import logger
from fleetx.optim import lr_scheduler as lr
from fleetx.core.module.basic_module import BasicModule
from fleetx.utils.tensor_fusion_helper import fused_parameters
from fleetx.models.vit_model import modeling as vit_model


class TopkAcc(nn.Layer):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]

        if len(label.shape) == 1:
            label = label.reshape([label.shape[0], -1])

        if label.dtype == paddle.int32:
            label = paddle.cast(label, 'int64')
        metric_dict = dict()
        for i, k in enumerate(self.topk):
            acc = paddle.metric.accuracy(x, label, k=k).item()
            metric_dict["top{}".format(k)] = acc
            if i == 0:
                metric_dict["metric"] = acc

        return metric_dict


class ViTModule(BasicModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        model_config = copy.deepcopy(configs['Model'])
        model_config_name = model_config.pop('name')

        self.model = getattr(vit_model, model_config_name)(**model_config)
        self.loss_fn = vit_model.ViTCELoss(epsilon=0.0001)
        self.eval_loss_fn = vit_model.CELoss()
        self.eval_metric_func = TopkAcc()
        self.train_batch_size = None
        self.eval_batch_size = None
        self.acc_list = []

        logger.info(f'Total parameters: {len(self.model.parameters())}')

    def configure_optimizers(self):
        self.decay_fused_tensors, self.all_fused_tensors = None, None

        opt_configs = self.configs['Optimizer']

        lr_configs = copy.deepcopy(self.configs['Optimizer']['lr'])
        lr_scheduler = lr.ViTLRScheduler(**lr_configs)

        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=opt_configs[
            'grad_clip']) if 'grad_clip' in opt_configs and opt_configs[
                'grad_clip'] > 0 else None

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.

        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            beta1=opt_configs['adam_beta1'],
            beta2=opt_configs['adam_beta2'],
            epsilon=opt_configs['adam_epsilon'],
            parameters=self.all_fused_tensors
            if self.configs['Fused']['tensor_fusion'] else
            self.model.parameters(),
            weight_decay=opt_configs['weight_decay'],
            grad_clip=clip,
            multi_precision=self.configs['Engine']['mix_precision'][
                'use_pure_fp16'])
        return optimizer, lr_scheduler

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        inputs, labels = batch

        if self.train_batch_size is None:
            self.train_batch_size = inputs.shape[
                0] * paddle.distributed.get_world_size()

        inputs.stop_gradient = True
        labels.stop_gradient = True

        logits = self(inputs)
        loss = self.loss_fn(logits, labels)

        return loss

    def training_step_end(self, log_dict):
        ips = self.train_batch_size * self.configs['Engine'][
            'logging_freq'] / log_dict['train_cost']

        logger.info(
            "[train] global step %d, epoch: %d, step: %d, learning rate: %.5e, loss: %.9f, batch_cost: %.5f sec, ips: %.2f images/sec"
            % (self.global_step, log_dict['epoch'], log_dict['batch'],
               self.optimizer.get_lr(), log_dict['loss'],
               log_dict['train_cost'], ips))

    def validation_step(self, batch):
        inputs, labels = batch

        batch_size = inputs.shape[0]

        inputs.stop_gradient = True
        labels.stop_gradient = True

        logits = self(inputs)
        loss = self.eval_loss_fn(logits, labels)

        if paddle.distributed.get_world_size() > 1:
            label_list = []
            paddle.distributed.all_gather(label_list, labels)
            labels = paddle.concat(label_list, 0)

            pred_list = []
            paddle.distributed.all_gather(pred_list, logits)
            logits = paddle.concat(pred_list, 0)

        if self.eval_batch_size is None:
            self.eval_batch_size = logits.shape[0]

        acc = self.eval_metric_func(logits, labels)
        self.acc_list.append(acc)
        return loss

    def validation_step_end(self, log_dict):
        ips = self.eval_batch_size * self.configs['Engine'][
            'logging_freq'] / log_dict['eval_cost']
        speed = self.configs['Engine']['logging_freq'] / log_dict['eval_cost']
        logger.info(
            "[eval] epoch: %d, step: %d, loss: %.9f, batch_cost: %.5f sec, ips: %.2f images/sec"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'],
               log_dict['eval_cost'], ips))
