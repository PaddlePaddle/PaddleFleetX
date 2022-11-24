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

import os
import sys
import copy
import datetime
from collections import defaultdict
import numpy as np

import paddle
import paddle.nn as nn
from ppfleetx.utils.log import logger

from ppfleetx.core.module.basic_module import BasicModule

from .factory import build
from .moco import MoCo


class MOCOModule(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()
        self.model_configs = copy.deepcopy(configs.Model)
        self.model_configs.pop('module')

        # must init before loss function
        super(MOCOModule, self).__init__(configs)

        assert 'train' in self.model_configs.loss
        self.loss_fn = build(self.model_configs.loss.train)

        self.train_batch_size = None
        self.best_metric = 0.0

    def get_model(self):
        if not hasattr(self, 'model') or self.model is None:
            config = copy.deepcopy(self.model_configs.model)
            base_encoder = build(self.model_configs.model.base_encoder)
            base_projector = build(
                self.model_configs.model.get('base_projector',
                                             {"name": "Identity"}))
            base_classifier = build(self.model_configs.model.base_classifier)
            momentum_encoder = build(self.model_configs.model.momentum_encoder)
            momentum_projector = build(
                self.model_configs.model.get('momentum_projector',
                                             {"name": "Identity"}))
            momentum_classifier = build(
                self.model_configs.model.momentum_classifier)

            config['base_encoder'] = base_encoder
            config['base_projector'] = base_projector
            config['base_classifier'] = base_classifier
            config['momentum_encoder'] = momentum_encoder
            config['momentum_projector'] = momentum_projector
            config['momentum_classifier'] = momentum_classifier

            self.model = MoCo(**config)
        return self.model

    def forward(self, img_q, img_k):
        return self.model(img_q, img_k)

    def training_step(self, batch):
        img_q, img_k = batch

        # Note(GuoxiaWang)paddle.distributed.all_gather required CudaPlace
        img_q = img_q.cuda()
        img_k = img_k.cuda()

        if self.train_batch_size is None:
            self.train_batch_size = img_q.shape[
                0] * paddle.distributed.get_world_size()

        logits, labels = self(img_q, img_k)
        loss = self.loss_fn(logits, labels)

        return loss

    def training_step_end(self, log_dict):
        ips = self.train_batch_size / log_dict['train_cost']

        total_step = log_dict['total_epoch'] * log_dict['total_batch']
        cur_step = log_dict['epoch'] * log_dict['total_batch'] + log_dict[
            'batch'] + 1
        remained_step = total_step - cur_step
        eta_sec = remained_step * log_dict['train_cost']
        eta_msg = "eta: {:s}".format(
            str(datetime.timedelta(seconds=int(eta_sec))))

        logger.info(
            "[train] epoch: %d, step: [%d/%d], learning rate: %.7f, loss: %.9f, batch_cost: %.5f sec, ips: %.2f images/sec, %s"
            % (log_dict['epoch'], log_dict['batch'], log_dict['total_batch'],
               log_dict['lr'], log_dict['loss'], log_dict['train_cost'], ips,
               eta_msg))

    def input_spec(self):
        return [
            InputSpec(
                shape=[None, 3, 224, 224], name="images", dtype='float32')
        ]

    def training_epoch_end(self, log_dict):
        logger.info("[Training] epoch: %d, total time: %.5f sec" %
                    (log_dict['epoch'], log_dict['train_cost']))


class MOCOClsModule(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()
        self.model_configs = copy.deepcopy(configs.Model)
        self.model_configs.pop('module')

        # must init before loss function
        super(MOCOClsModule, self).__init__(configs)

        assert 'train' in self.model_configs.loss
        self.loss_fn = build(self.model_configs.loss.train)
        self.eval_loss_fn = None
        if 'eval' in self.model_configs.loss:
            self.eval_loss_fn = build(self.model_configs.loss.eval)

        if 'train' in self.model_configs.metric:
            self.train_metric_fn = build(self.model_configs.metric.train)
        if 'eval' in self.model_configs.metric:
            self.eval_metric_fn = build(self.model_configs.metric.eval)

        self.train_batch_size = None
        self.eval_batch_size = None
        self.best_metric = 0.0
        self.acc_list = []

    def _freeze_backbone(self, layer):
        for param in layer.parameters():
            param.trainable = False

        def freeze_norm(layer):
            if isinstance(layer, (nn.layer.norm._BatchNormBase)):
                layer._use_global_stats = True

        layer.apply(freeze_norm)

    def get_model(self):
        if not hasattr(self, 'model') or self.model is None:
            pretrained_path = self.model_configs.model.base_encoder.pop(
                "pretrained")
            base_encoder = build(self.model_configs.model.base_encoder)
            self._freeze_backbone(base_encoder)

            pretrained_path = pretrained_path + ".pdparams"
            assert os.path.exists(
                pretrained_path), f'{pretrained_path} is not exists!'
            base_encoder_dict = paddle.load(pretrained_path)

            for k in list(base_encoder_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('base_encoder.0.'):
                    # remove prefix
                    base_encoder_dict[k[len(
                        "base_encoder.0."):]] = base_encoder_dict[k]
                    # delete renamed
                    del base_encoder_dict[k]

            for name, param in base_encoder.state_dict().items():
                if name in base_encoder_dict and param.dtype != base_encoder_dict[
                        name].dtype:
                    base_encoder_dict[name] = base_encoder_dict[name].cast(
                        param.dtype)

            base_encoder.set_state_dict(base_encoder_dict)
            logger.info(f'Load pretrained weight from {pretrained_path}')

            base_classifier = build(self.model_configs.model.base_classifier)

            self.model = nn.Sequential(base_encoder, base_classifier)
        return self.model

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
        ips = self.train_batch_size / log_dict['train_cost']

        total_step = log_dict['total_epoch'] * log_dict['total_batch']
        cur_step = log_dict['epoch'] * log_dict['total_batch'] + log_dict[
            'batch'] + 1
        remained_step = total_step - cur_step
        eta_sec = remained_step * log_dict['train_cost']
        eta_msg = "eta: {:s}".format(
            str(datetime.timedelta(seconds=int(eta_sec))))

        logger.info(
            "[train] epoch: %d, step: [%d/%d], learning rate: %.7f, loss: %.9f, batch_cost: %.5f sec, ips: %.2f images/sec, %s"
            % (log_dict['epoch'], log_dict['batch'], log_dict['total_batch'],
               log_dict['lr'], log_dict['loss'], log_dict['train_cost'], ips,
               eta_msg))

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

        acc = self.eval_metric_fn(logits, labels)
        self.acc_list.append(acc)
        return loss

    def validation_step_end(self, log_dict):
        ips = self.eval_batch_size / log_dict['eval_cost']
        speed = self.configs['Engine']['logging_freq'] / log_dict['eval_cost']
        logger.info(
            "[eval] epoch: %d, step: [%d/%d], loss: %.9f, batch_cost: %.5f sec, ips: %.2f images/sec"
            % (log_dict['epoch'], log_dict['batch'], log_dict['total_batch'],
               log_dict['loss'], log_dict['eval_cost'], ips))

    def input_spec(self):
        return [
            InputSpec(
                shape=[None, 3, 224, 224], name="images", dtype='float32')
        ]

    def training_epoch_end(self, log_dict):
        logger.info("[Training] epoch: %d, total time: %.5f sec" %
                    (log_dict['epoch'], log_dict['train_cost']))

    def validation_epoch_end(self, log_dict):
        msg = ''
        if len(self.acc_list) > 0:
            ret = defaultdict(list)

            for item in self.acc_list:
                for key, val in item.items():
                    ret[key].append(val)

            for k, v in ret.items():
                ret[k] = np.mean(v)

            if 'metric' in ret and ret['metric'] > self.best_metric:
                self.best_metric = ret['metric']

            if 'metric' in ret:
                ret['best_metric'] = self.best_metric

            msg = ', '
            msg += ", ".join([f'{k} = {v:.6f}' for k, v in ret.items()])
            self.acc_list.clear()

        logger.info("[Eval] epoch: %d, total time: %.5f sec%s" %
                    (log_dict['epoch'], log_dict['eval_cost'], msg))
