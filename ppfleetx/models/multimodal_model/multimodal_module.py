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

from ppfleetx.core.module.basic_module import BasicModule
import ppfleetx.models.multimodal_model.imagen as imagen
from ppfleetx.utils.log import logger

from .utils import process_configs


class MultiModalModule(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()
        super(MultiModalModule, self).__init__(configs)

        self.loss_fn = self.get_loss_fn()

    def process_configs(self, configs):
        configs = process_configs(configs)
        return configs

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch):
        preds, targets, log_snr, p2_loss_weight_gamma = self(batch)
        loss = self.loss_fn(preds, targets, log_snr, p2_loss_weight_gamma)
        return loss

    def training_step_end(self, log_dict):
        speed = self.configs.Engine.logging_freq / log_dict['train_cost']

        logger.info(
            "[train] epoch: %d, batch: %d, loss: %.9f, avg_batch_cost: %.5f sec, speed: %.2f step/s, learning rate: %.5e"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'],
               1. / speed, speed, log_dict['lr']))

    def validation_step(self, batch):
        tokens, position_ids, labels, loss_mask = batch
        preds = self(tokens, position_ids)
        preds = paddle.cast(preds, dtype="float32")
        loss = self.loss_fn(preds, labels, loss_mask)
        return loss

    def validation_step_end(self, log_dict):
        speed = self.configs.Engine.logging_freq / log_dict['eval_cost']
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
        speed = self.configs.Engine.logging_freq / log_dict['test_cost']
        logger.info(
            "[test] epoch: %d, batch: %d, loss: %.9f, avg_test_cost: %.5f sec, speed: %.2f step/s"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'],
               1. / speed, speed))

    def input_spec(self):
        return [
            InputSpec(
                shape=[None, None], name="tokens", dtype='int64'), InputSpec(
                    shape=[None, None], name="ids", dtype='int64')
        ]

    def training_epoch_end(self, log_dict):
        logger.info("[Training] epoch: %d, total time: %.5f sec" %
                    (log_dict['epoch'], log_dict['train_cost']))


class ImagenModule(MultiModalModule):
    def __init__(self, configs):
        super(ImagenModule, self).__init__(configs)

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        imagen_model = model_setting.pop("name")
        model = getattr(imagen, imagen_model)(**model_setting)
        return model

    def get_loss_fn(self):
        model_setting = copy.deepcopy(self.configs.Loss)
        loss_fn = imagen.ImagenCriterion(**model_setting)
        return loss_fn

    def pretreating_batch(self, batch):
        return batch
