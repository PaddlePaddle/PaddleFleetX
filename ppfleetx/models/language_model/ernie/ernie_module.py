#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import ppfleetx.models.language_model.gpt as gpt
from ppfleetx.utils.log import logger

from .dygraph.single_model import ErnieModel, ErnieForPretraining, ErniePretrainingCriterion
from .dygraph.hybrid_model import ErnieModelHybrid, ErnieForPretrainingHybrid, ErniePretrainingCriterionHybrid, ErnieForPretrainingPipe

from ppfleetx.models.language_model.utils import process_configs

import numpy as np


def process_data_configs(config):
    """
    process data configs for hybrid parallel
    """
    cfg_global = config['Global']
    cfg_data = config['Data']

    mode_to_num_samples = {
        "Train":
        cfg_global['global_batch_size'] * config['Engine']['max_steps'],
        "Eval": cfg_global['global_batch_size'] *
        (config['Engine']['max_steps'] // config['Engine']['eval_freq'] + 1) *
        config['Engine']['eval_iters'],
        "Test":
        cfg_global['global_batch_size'] * config['Engine']['test_iters'],
    }

    for mode in ("Train", "Eval", "Test"):
        if mode in cfg_data.keys():
            cfg_data[mode]['dataset']['num_samples'] = mode_to_num_samples[
                mode]
            cfg_data[mode]['dataset']['mode'] = mode
            cfg_data[mode]['dataset']['seed'] = cfg_global['seed']
            cfg_data[mode]['sampler']['batch_size'] = cfg_global[
                'local_batch_size']
            cfg_data[mode]['dataset'].setdefault('binary_head',
                                                 cfg_global['binary_head'])
            cfg_data[mode]['loader']['collate_fn'].setdefault(
                'micro_batch_size', cfg_global['micro_batch_size'])


def process_model_configs(config):
    cfg_model = config['Model']
    hidden_size = cfg_model['hidden_size']
    cfg_model.setdefault("intermediate_size", hidden_size * 4)


class ErnieModule(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()
        super(ErnieModule, self).__init__(configs)
        self.nranks = paddle.distributed.get_world_size()
        self.binary_head = self.configs['Global']['binary_head']

        if self.nranks > 1:
            self.criterion = ErniePretrainingCriterionHybrid(self.binary_head)
        else:
            self.criterion = ErniePretrainingCriterion(self.binary_head)

    def process_configs(self, configs):
        process_data_configs(configs)
        process_model_configs(configs)
        return configs

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        if self.nranks > 1:
            model_setting[
                'num_partitions'] = self.configs.Distributed.mp_degree
            # model = ErnieForPretrainingHybrid(ErnieModelHybrid(**model_setting))

            if self.configs.Distributed.pp_degree == 1:
                model = ErnieForPretrainingHybrid(
                    ErnieModelHybrid(**model_setting))
            else:
                model = ErnieForPretrainingPipe(**model_setting)
        else:
            model = ErnieForPretraining(ErnieModel(**model_setting))

        return model

    def forward(self, tokens):
        return self.model(tokens)

    def pretreating_batch(self, batch):
        if self.configs.Distributed.pp_degree > 1:
            input_ids, segment_ids, input_mask, masked_lm_positions, \
                        masked_lm_labels, next_sentence_labels = batch
            data = [
                (input_ids, segment_ids, input_mask),
                (masked_lm_positions, masked_lm_labels, next_sentence_labels)
            ]
            return data
        else:
            return batch

    def training_step(self, batch):
        input_ids, segment_ids, input_mask, masked_lm_positions, \
            masked_lm_labels, next_sentence_labels = batch

        # Create the model for the ernie pretrain
        if self.binary_head:
            prediction_scores, seq_relationship_score = self.model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                # position_ids=None,
                attention_mask=input_mask,
                masked_positions=masked_lm_positions)
            lm_loss, sop_loss = self.criterion(
                prediction_scores, seq_relationship_score, masked_lm_labels,
                next_sentence_labels)
            loss = lm_loss + sop_loss
        else:
            prediction_scores = self.model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                # position_ids=None,
                attention_mask=input_mask,
                masked_positions=masked_lm_positions)

            loss = self.criterion(prediction_scores, None, masked_lm_labels)

        return loss

    def training_step_end(self, log_dict):
        speed = 1. / log_dict['train_cost']
        default_global_tokens_num = self.configs.Global.global_batch_size * \
            self.configs.Data.Train.dataset.max_seq_length

        logger.info(
            "[train] epoch: %d, batch: %d, loss: %.9f, avg_batch_cost: %.5f sec, speed: %.2f step/s, " \
            "ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'], log_dict['train_cost'], speed,
               speed * default_global_tokens_num, speed * default_global_tokens_num / self.nranks, log_dict['lr']))
