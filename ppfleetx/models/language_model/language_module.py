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
import types

import paddle
import paddle.distributed.fleet as fleet

sys.path.append("../../../../")
from ppfleetx.core.module.basic_module import BasicModule
import ppfleetx.models.language_model.gpt as gpt
from ppfleetx.utils import logger
import paddleslim
from .utils import process_configs


class LanguageModule(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()

        super(LanguageModule, self).__init__(configs)

    def process_configs(self, configs):
        configs = process_configs(configs)
        return configs

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
        speed = 1. / log_dict['train_cost']
        default_global_tokens_num = self.configs.Global.global_batch_size * \
            self.configs.Data.Train.dataset.max_seq_len

        logger.info(
            "[train] epoch: %d, batch: %d, loss: %.9f, avg_batch_cost: %.5f sec, speed: %.2f step/s, " \
            "ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'], log_dict['train_cost'], speed,
               speed * default_global_tokens_num, speed * default_global_tokens_num / self.nranks, log_dict['lr']))

    def training_epoch_end(self, log_dict):
        logger.info("[Training] epoch: %d, total time: %.5f sec" %
                    (log_dict['epoch'], log_dict['train_cost']))

    def validation_step(self, batch):
        tokens, position_ids, labels, loss_mask = batch
        preds = self(tokens, position_ids)
        preds = paddle.cast(preds, dtype="float32")
        loss = self.loss_fn(preds, labels, loss_mask)
        return loss

    def validation_step_end(self, log_dict):
        speed = 1. / log_dict['eval_cost']
        logger.info(
            "[eval] epoch: %d, batch: %d, loss: %.9f, avg_eval_cost: %.5f sec, speed: %.2f step/s"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'],
               log_dict['eval_cost'], speed))

    def test_step(self, batch):
        tokens, position_ids, labels, loss_mask = batch
        preds = self(tokens, position_ids)
        preds = paddle.cast(preds, dtype="float32")
        loss = self.loss_fn(preds, labels, loss_mask)
        return loss

    def test_step_end(self, log_dict):
        speed = 1. / log_dict['test_cost']
        logger.info(
            "[test] epoch: %d, batch: %d, loss: %.9f, avg_test_cost: %.5f sec, speed: %.2f step/s"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'],
               log_dict['test_cost'], speed))

    def input_spec(self):
        return [
            InputSpec(
                shape=[None, None], name="tokens", dtype='int64'), InputSpec(
                    shape=[None, None], name="ids", dtype='int64')
        ]


class GPTModule(LanguageModule):
    def __init__(self, configs):
        super(GPTModule, self).__init__(configs)

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        moe_configs = model_setting.get('moe_configs', {'expert_mode': False})
        assert not moe_configs[
            'expert_mode'], "Not support expert mode in GPT model!"

        if not moe_configs['expert_mode']:
            model_setting['moe_configs'] = moe_configs

        l = model_setting['num_layers']
        h = model_setting['hidden_size']
        v = model_setting['vocab_size']
        s = self.configs.Data.Train.dataset.max_seq_len
        self.get_model_size(l, h, v, s)

        if self.nranks == 1:
            model = gpt.GPTForPretraining(gpt.GPTModel(**model_setting))
        else:
            if self.configs.Distributed.pp_degree == 1:
                model = gpt.GPTForPretrainingHybrid(
                    gpt.GPTModelHybrid(**model_setting))
            else:
                model = gpt.GPTForPretrainingPipe(**model_setting)

        return model

    def get_loss_fn(self):
        if self.nranks == 1:
            loss_fn = gpt.GPTPretrainingCriterion()
        else:
            loss_fn = gpt.GPTPretrainingCriterionHybird()
        return loss_fn

    def get_model_size(self, l, h, v, s):
        P = 12 * l * h * h * (1 + 13 / (12 * h) + (v + s) / (12 * l * h))
        logger.info('Model Size: {:.2f} B'.format(P / 1000.0 / 1000.0 /
                                                  1000.0))

    def pretreating_batch(self, batch):
        if self.configs.Distributed.pp_degree > 1:
            tokens, position_ids, labels, loss_mask = batch
            data = [(tokens, position_ids), (labels, loss_mask)]
            return data
        else:
            return batch

    def qat_model(self):
        quanter = paddleslim.dygraph.quant.QAT(
            config=self.configs.Quantization)
        self.model = quanter.quantize(self.model)


class MoEModule(LanguageModule):
    def __init__(self, configs):
        self.initialize_model_and_expert_group()
        super(MoEModule, self).__init__(configs)

        assert self.nranks == configs.Distributed.dp_degree, \
            "only support single card or data parallel in MoE model."

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        l = model_setting['num_layers']
        h = model_setting['hidden_size']
        v = model_setting['vocab_size']
        s = self.configs.Data.Train.dataset.max_seq_len
        self.get_model_size(l, h, v, s)

        model = gpt.GPTForPretrainingHybrid(
            gpt.GPTModelHybrid(**model_setting))

        return model

    def get_loss_fn(self):
        return gpt.GPTPretrainingCriterionHybird()

    def get_model_size(self, l, h, v, s):
        P = 12 * l * h * h * (1 + 13 / (12 * h) + (v + s) / (12 * l * h))
        logger.info('Model Size: {:.2f} B'.format(P / 1000.0 / 1000.0 /
                                                  1000.0))

    def training_step(self, batch):
        tokens, position_ids, labels, loss_mask = batch

        loss_mask.stop_gradient = True
        labels.stop_gradient = True
        position_ids.stop_gradient = True

        preds = self(tokens, position_ids)
        loss = self.loss_fn(preds, labels, loss_mask)

        with paddle.amp.auto_cast(enable=False):
            if self.configs.Model.moe_configs.gate != "naive" and \
                self.configs.Engine.balance_loss_weight:

                gpt_layer = self.model._layers.gpt if isinstance(
                    self.model, paddle.DataParallel) else self.model.gpt

                aux_loss_list = [
                    l.moe_mlp.gate.get_loss(clear=False)
                    for l in gpt_layer.decoder.layers
                    if hasattr(l.moe_mlp, "gate")
                ]
                bal_loss = paddle.concat(aux_loss_list)
                if bal_loss.dtype == paddle.float16:
                    bal_loss = paddle.cast(bal_loss, dtype=paddle.float32)
                bal_loss = bal_loss.mean()
                loss += bal_loss * self.configs.Engine.balance_loss_weight

        return loss

    def initialize_model_and_expert_group(self):
        hcg = fleet.get_hybrid_communicate_group()

        def get_expert_parallel_world_size(self):
            return self.get_data_parallel_world_size(
            ) * self.get_model_parallel_world_size()

        hcg.get_expert_parallel_world_size = types.MethodType(
            get_expert_parallel_world_size, hcg)

        # need create mp_dp group for expert parallel group in advance
        _, mp_dp_comm_group = hcg._set_check_group(parallel_method="pipe")

        def get_expert_parallel_group(self):
            return mp_dp_comm_group

        hcg.get_expert_parallel_group = types.MethodType(
            get_expert_parallel_group, hcg)

    def initialize_mp_dp_parameters(self):
        hcg = fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        mp_src_rank = hcg.get_model_parallel_group_src_rank()

        dp_group = hcg.get_data_parallel_group()
        dp_src_rank = hcg.get_data_parallel_group_src_rank()

        for param in self.model.parameters():
            if "expert_" in param.name:
                setattr(param, "no_sync", True)
                continue

            if not param.is_distributed:
                paddle.distributed.broadcast(
                    param.detach(),
                    src=mp_src_rank,
                    group=mp_group,
                    use_calc_stream=True)

            paddle.distributed.broadcast(
                param.detach(),
                src=dp_src_rank,
                group=dp_group,
                use_calc_stream=True)
