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
import yaml
import codecs
from collections.abc import Mapping

import paddle
from paddle.static import InputSpec
import paddle.nn as nn

from ppfleetx.core.module.basic_module import BasicModule
import ppfleetx.models.language_model.gpt as gpt
from ppfleetx.utils.log import logger

from .dygraph.single_model import (
    ErnieModel,
    ErnieForPretraining,
    ErniePretrainingCriterion,
    ErnieForSequenceClassification, )
from .dygraph.hybrid_model import (ErnieModelHybrid, ErnieForPretrainingHybrid,
                                   ErniePretrainingCriterionHybrid,
                                   ErnieForPretrainingPipe,
                                   ErnieForSequenceClassificationHybrid)

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


def process_finetune_configs(task, config):
    cfg_data = config['Data']
    cfg_dist = config['Distributed']
    cfg_optim = config['Optimizer']
    cfg_global = config['Global']
    cfg_engine = config['Engine']

    path = "./ppfleetx/models/language_model/ernie/finetune_configs.yaml"
    with codecs.open(path, 'r', 'utf-8') as file:
        dic = yaml.load(file, Loader=yaml.FullLoader)

    dataset_type = cfg_data.Train.dataset.dataset_type
    assert dataset_type in dic[task].keys(
    ), "{} is an invalid dataset type ! Only support the types of dataset shown in {}".format(
        dataset_type, path)

    num_train_epochs = dic[task][dataset_type].get('num_train_epochs', None)
    if num_train_epochs is not None:
        cfg_engine['num_train_epochs'] = num_train_epochs

    learning_rate = dic[task][dataset_type].get("learning_rate", None)
    if learning_rate is not None:
        cfg_optim['lr']['max_lr'] = learning_rate

    max_seq_length = dic[task][dataset_type].get("max_seq_length", None)
    if max_seq_length is not None:
        for mode in ("Train", "Eval", "Test"):
            if mode in cfg_data.keys():
                cfg_data[mode]['dataset']['max_seq_len'] = max_seq_length

    batch_size = dic[task][dataset_type].get("batch_size", None)
    if batch_size is not None:
        assert batch_size % cfg_global['micro_batch_size'] == 0

        cfg_global['local_batch_size'] = batch_size
        cfg_global['global_batch_size'] = batch_size * cfg_dist[
            'dp_degree'] * cfg_dist['pp_degree']


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

    def get_model_size(self, l, h, v, s):
        P = 12 * l * h * h * (1 + 13 / (12 * h) + (v + s) / (12 * l * h))
        logger.info('Model Size: {:.2f} B'.format(P / 1000.0 / 1000.0 /
                                                  1000.0))

    def process_configs(self, configs):
        process_data_configs(configs)
        process_model_configs(configs)
        return configs

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        l = model_setting['num_hidden_layers']
        h = model_setting['hidden_size']
        v = model_setting['vocab_size']
        s = self.configs.Data.Train.dataset.max_seq_length
        self.get_model_size(l, h, v, s)

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

            if not isinstance(masked_lm_positions, list):
                masked_lm_positions = [masked_lm_positions]
            if not isinstance(masked_lm_labels, list):
                masked_lm_labels = [masked_lm_labels]

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

    def input_spec(self):
        return [
            InputSpec(
                shape=[None, None], dtype='int64'), InputSpec(
                    shape=[None, None], dtype='int64'), InputSpec(
                        shape=[None, None], dtype='int64')
        ]


class ErnieSeqClsModule(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()
        super(ErnieSeqClsModule, self).__init__(configs)

        self.criterion = nn.loss.CrossEntropyLoss(
        )  # if data_args.label_list else nn.loss.MSELoss()

        self.past_index = -1
        self.past = None
        self.label_names = (["start_positions", "end_positions"] \
            if "QusetionAnswering" in type(self.model).__name__ else ["labels"])

    def process_configs(self, configs):
        process_model_configs(configs)
        process_finetune_configs("SequenceClassification", configs)

        cfg_global = configs['Global']
        cfg_data = configs['Data']

        for mode in ("Train", "Eval", "Test"):
            if mode in cfg_data.keys():
                cfg_data[mode]['dataset']['mode'] = mode
                cfg_data[mode]['sampler']['batch_size'] = cfg_global[
                    'local_batch_size']
                cfg_data[mode]['loader']['collate_fn'].setdefault(
                    'tokenizer_type',
                    cfg_data[mode]['dataset']['tokenizer_type'])

        return configs

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        if self.nranks > 1:
            model_setting[
                'num_partitions'] = self.configs.Distributed.mp_degree

            if self.configs.Distributed.pp_degree == 1:
                model = ErnieForSequenceClassificationHybrid(
                    ErnieModelHybrid(**model_setting))
            else:
                raise ValueError(
                    "Pipeline Parallelism is not supported in Sequence \
                    Classification task of Ernie model.")
        else:
            model = ErnieForSequenceClassification(ErnieModel(**model_setting))

        return model

    def prepare_input(self, data):
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)(
                {k: self.prepare_input(v)
                 for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self.prepare_input(v) for v in data)
        elif isinstance(data, paddle.Tensor):
            # kwargs = dict(device=self.args.current_device)
            # update data type for pure fp16
            return data
            # return data.to(**kwargs)
        return data

    def pretreating_batch(self, batch):
        self.has_labels = all(
            batch.get(k) is not None for k in self.label_names)

        batch = self.prepare_input(batch)
        if self.past_index >= 0 and self.past is not None:
            batch["mems"] = self.past

        return batch

    def forward(self, inputs):
        return self.model(**inputs)

    def compute_loss(self, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        elif "start_positions" in inputs and "end_positions" in inputs:
            labels = (inputs.pop("start_positions"),
                      inputs.pop("end_positions"))
        elif "generator_labels" in inputs:
            labels = inputs["generator_labels"]
        else:
            labels = None
        outputs = self(inputs)

        loss = self.criterion(outputs, labels)
        outputs = (loss, outputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.past_index >= 0:
            self.past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, batch):
        return self.compute_loss(batch)

    def training_step_end(self, log_dict):
        speed = 1. / log_dict['train_cost']
        default_global_tokens_num = self.configs.Global.global_batch_size * \
            self.configs.Data.Train.dataset.max_seq_len

        logger.info(
            "[train] epoch: %d, batch: %d, loss: %.9f, avg_batch_cost: %.5f sec, speed: %.2f step/s, " \
            "ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'], log_dict['train_cost'], speed,
               speed * default_global_tokens_num, speed * default_global_tokens_num / self.nranks, log_dict['lr']))

    def input_spec(self):
        input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64")  # segment_ids
        ]
        return input_spec

    def validation_step(self, inputs):
        if self.has_labels:
            loss, outputs = self.compute_loss(inputs, return_outputs=True)
            loss = loss.mean().detach()

        else:
            loss = None

        return loss

    def validation_step_end(self, log_dict):
        speed = 1. / log_dict['eval_cost']
        logger.info(
            "[eval] epoch: %d, batch: %d, loss: %.9f, avg_eval_cost: %.5f sec, speed: %.2f step/s"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'],
               log_dict['eval_cost'], speed))
