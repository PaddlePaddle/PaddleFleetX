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

sys.path.append("../../../../")
from ppfleetx.core.module.basic_module import BasicModule
import ppfleetx.models.language_model.gpt as gpt
import paddleslim


class LanguageModule(BasicModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.loss_fn = self.get_loss_fn()

    def get_loss_fn(self):
        raise NotImplementedError

    def forward(self, tokens, ids):
        return self.model(tokens, ids)

    def training_step(self, batch):
        tokens, position_ids, labels, loss_mask = batch

        loss_mask.stop_gradient = True
        labels.stop_gradient = True
        position_ids.stop_gradient = True

        preds = self(tokens, position_ids)
        loss = self.loss_fn(preds, labels, loss_mask)

        return loss

    def training_step_end(self, log_dict):
        speed = self.configs.Engine.logging_freq / log_dict['train_cost']
        default_global_tokens_num = self.configs.Global.global_batch_size * \
            self.configs.Global.max_seq_len

        logger.info(
            "[train] global step %d, epoch: %d, batch: %d, loss: %.9f, avg_batch_cost: %.5f sec, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s"
            % (self.global_step, log_dict['epoch'], log_dict['batch'],
               log_dict['loss'], 1. / speed, speed, speed *
               default_global_tokens_num, speed * default_global_tokens_num))

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

    def qat_model(self):
        quanter = paddleslim.dygraph.quant.QAT(
            config=self.configs.Quantization)
        self.model = quanter.quantize(self.model)

    def input_spec(self):
        return [
            InputSpec(
                shape=[None, None], name="tokens", dtype='int64'), InputSpec(
                    shape=[None, None], name="ids", dtype='int64')
        ]


class GPTModule(LanguageModule):
    def __init__(self, configs):
        super().__init__(configs)
        self.nranks = paddle.distributed.get_world_size()

    def get_model(self):
        model_setting = self.configs.Model
        if self.nranks == 1:
            model = gpt.GPTForPretraining(gpt.GPTModel(model_setting))
        else:
            if self.configs.Distributed.pp_degree == 1:
                model = gpt.GPTForPretrainingHybrid(
                    gpt.GPTModelHybrid(model_setting))
            else:
                model = gpt.GPTForPretrainingPipe(model_setting)
        return model

    def get_loss_fn(self):
        if self.nranks == 1:
            loss_fn = gpt.GPTPretrainingCriterion()
        else:
            loss_fn = gpt.GPTPretrainingCriterionHybird()
        return loss_fn

    def pretreating_batch(self, batch):
        if self.configs.Distributed.pp_degree > 1:
            tokens, position_ids, labels, loss_mask = batch
            data = [(tokens, position_ids), (labels, loss_mask)]
            return data
        else:
            return batch
