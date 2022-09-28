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

from .single_model import ErnieModel, ErnieForPretraining, ErniePretrainingCriterion
from ppfleetx.models.language_model.utils import process_configs


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
    # return cfg


class ErnieModule(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()
        super(ErnieModule, self).__init__(configs)
        self.criterion = ErniePretrainingCriterion(False)

    def process_configs(self, configs):
        process_data_configs(configs)
        return configs

    def get_loss_fn(self):
        return None

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")
        model = ErnieForPretraining(ErnieModel(**model_setting))
        return model

    def forward(self, tokens):
        return self.model(tokens)

    def training_step(self, batch):
        tokens, position_ids, labels, loss_mask = batch

        prediction_scores, seq_relationship_score = self(tokens)
        prediction_scores.reshape_([-1, 50304])
        s = prediction_scores.shape
        loss_mask = paddle.randint(low=0, high=50304, shape=[s[0], 1])
        loss = self.criterion(
            prediction_scores,
            seq_relationship_score,
            loss_mask,
            next_sentence_labels=None)
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
