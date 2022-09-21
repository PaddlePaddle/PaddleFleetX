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
import math

import paddle
from paddle.static import InputSpec

from ppfleetx.core.module.basic_module import BasicModule
import ppfleetx.models.language_model.gpt as gpt
from ppfleetx.utils import env
from ppfleetx.utils.log import logger
import paddleslim
from .utils import process_configs
from ppfleetx.data.tokenizers import GPTTokenizer


class LanguageModule(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()
        self.data_world_size = env.get_data_world_size()
        super(LanguageModule, self).__init__(configs)

        self.loss_fn = self.get_loss_fn()

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
               speed * default_global_tokens_num, speed * default_global_tokens_num / self.data_world_size, log_dict['lr']))

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

    def qat_model(self):
        quanter = paddleslim.dygraph.quant.QAT(
            config=self.configs.Quantization)
        return quanter.quantize(self.model)

    def get_model_size(self, l, h, v, s):
        P = 12 * l * h * h * (1 + 13 / (12 * h) + (v + s) / (12 * l * h))
        logger.info('Model Size: {:.2f} B'.format(P / 1000.0 / 1000.0 /
                                                  1000.0))

    def training_epoch_end(self, log_dict):
        logger.info("[Training] epoch: %d, total time: %.5f sec" %
                    (log_dict['epoch'], log_dict['train_cost']))


class GPTModule(LanguageModule):
    def __init__(self, configs):
        super(GPTModule, self).__init__(configs)

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        l = model_setting['num_layers']
        h = model_setting['hidden_size']
        v = model_setting['vocab_size']
        s = self.configs.Data.Train.dataset.max_seq_len
        self.get_model_size(l, h, v, s)

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        if self.nranks == 1:
            model_setting.pop("sequence_parallel")
            model = gpt.GPTForPretraining(gpt.GPTModel(**model_setting))
        else:
            model_setting[
                'num_partitions'] = self.configs.Distributed.mp_degree
            if self.configs.Distributed.pp_degree == 1:
                model_setting.pop("virtual_pp_degree", None)
                model = gpt.GPTForPretrainingHybrid(
                    gpt.GPTModelHybrid(**model_setting))
            else:
                model = gpt.GPTForPretrainingPipe(**model_setting)

        if 'Quantization' in self.configs.keys(
        ) and self.configs.Quantization.enable:
            model = self.qat_model(model)

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

    def input_spec(self):
        return [
            InputSpec(
                shape=[None, None], name="tokens", dtype='int64'), InputSpec(
                    shape=[None, None], name="ids", dtype='int64')
        ]

    def inference_end(self, outputs):
        for k, v in outputs.items():
            for i in range(v.shape[0]):
                out_ids = [int(x) for x in v[i]]
                ret_str = self.tokenizer.decode(out_ids)
                # ret_str = text[i] + ret_str
                print(ret_str)


class GPTGenerationModule(BasicModule):
    def __init__(self, configs):
        self.configs = configs
        self.generation_cfgs = configs.Generation
        self.nranks = paddle.distributed.get_world_size()

        super().__init__(configs)

    def process_configs(self, configs):
        configs = process_configs(configs)
        return configs

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        if self.nranks == 1:
            model = gpt.GPTForGeneration(
                gpt.GPTModel(**model_setting), self.generation_cfgs)
        else:
            assert self.nranks == self.configs.Distributed.dp_degree, \
                "only support single card and data parallel in generation task."
            model = gpt.GPTForGenerationHybrid(
                gpt.GPTModelHybrid(**model_setting), self.generation_cfgs)

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        self.generation_cfgs['max_dec_len'] = self.adjust_length_to_model(
            self.generation_cfgs['max_dec_len'], 512)

        self.generation_cfgs['bos_token_id'] = self.tokenizer.eos_token_id
        self.generation_cfgs['eos_token_id'] = self.tokenizer.eos_token_id
        self.generation_cfgs['pad_token_id'] = self.tokenizer.eos_token_id

        return model

    def adjust_length_to_model(self, length, max_sequence_length):
        if length < 0 or length > max_sequence_length:
            length = max_sequence_length
        return length

    def left_padding(self, inputs, pad_id, padding="longest"):
        assert "input_ids" in inputs, "input_ids should be in inputs!"
        max_length = 0
        for ids in inputs["input_ids"]:
            max_length = max(max_length, len(ids))

        def extend_max_lenth(value, max_length, to_pad_id):
            return [to_pad_id] * (max_length - len(value)) + value

        def extend_filed(name, max_length, to_pad_id):
            values = inputs[name]
            res = []
            for index, value in enumerate(values):
                res.append(extend_max_lenth(value, max_length, to_pad_id))
            inputs[name] = res

        extend_filed("input_ids", max_length, pad_id)
        if "attention_mask" in inputs:
            extend_filed("attention_mask", max_length, 0)
        if "position_ids" in inputs:
            extend_filed("position_ids", max_length, 0)

        return inputs

    def generate(self, input_text):
        return self(input_text)

    def forward(self, input_text):
        input_ids = self.tokenizer.encode(input_text)
        inputs = {'input_ids': [input_ids]}

        inputs = self.left_padding(inputs, self.tokenizer.eos_token_id)
        input_ids = inputs['input_ids']

        if len(input_ids) == 0:
            input_ids = None
        else:
            # [1, seq_len]
            input_ids = paddle.to_tensor(input_ids, dtype='int64')

        ids, scores = self.model(input_ids=input_ids)

        generated_sequences = []
        for i, generated_ids in enumerate(ids):
            generated_ids = generated_ids.numpy().tolist()
            # Decode text
            text = self.tokenizer.convert_ids_to_string(generated_ids)
            sequence = input_text + text
            generated_sequences.append(sequence)

        return generated_sequences

    def input_spec(self):
        return [InputSpec(shape=[None, None], name="input_ids", dtype='int64')]


class GPTEvalModule(LanguageModule):
    def __init__(self, configs):
        self.eval_cfgs = configs.Offline_Eval

        super().__init__(configs)

        self.post_process_configs()

        self.first_step = True
        self.total_score = 0
        self.score_name = "loss" if not self.eval_cfgs.cloze_eval else "number correct"

    def post_process_configs(self):
        self.configs.pop("Optimizer", None)
        self.configs.pop("Inference", None)

        self.configs.Data.pop("Train", None)
        self.configs.Data.pop("Test", None)
        self.configs.Data.Eval.pop("sampler", None)
        self.configs.Data.Eval.loader.collate_fn = "gpt_eval_collate_fn"
        self.configs.Data.Eval.loader.batch_size = self.eval_cfgs.batch_size
        self.configs.Data.Eval.dataset.input_dir = self.eval_cfgs.eval_path
        self.configs.Data.Eval.dataset.max_seq_len = self.eval_cfgs.max_seq_len

        self.configs.Engine.logging_freq = self.eval_cfgs.logging_freq

        if not self.eval_cfgs.cloze_eval:
            self.configs.Data.Eval.dataset.name = "LM_Eval_Dataset"
            self.configs.Data.Eval.dataset.overlapping_eval = self.eval_cfgs.overlapping_eval
        else:
            self.configs.Data.Eval.dataset.name = "Lambada_Eval_Dataset"

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")
        model_setting.pop("name")

        if self.nranks == 1:
            model = gpt.GPTForPretraining(gpt.GPTModel(**model_setting))
        else:
            raise RuntimeError(
                "Only single-card offline eval is supported in GPTModel now.")

        return model

    def forward(self, tokens, ids, mask):
        return self.model(tokens, ids, mask)

    def validation_step(self, batch):
        tokens, loss_mask, attention_mask, position_ids, labels, info = batch

        preds = self(tokens, position_ids, attention_mask)

        if not self.eval_cfgs.cloze_eval:
            if self.first_step:
                self.num_original_tokens = info.numpy()[0][0]
                self.num_tokenized_tokens = info.numpy()[0][1]

            masked_lm_loss = paddle.nn.functional.cross_entropy(
                preds, labels, reduction="none")
            loss = paddle.sum(masked_lm_loss * loss_mask)
            return loss
        else:
            if self.first_step:
                self.num_examples = info.numpy()[0][0]

            outputs = paddle.argmax(preds, -1)
            acc = paddle.cast(outputs == labels, 'float32')
            acc = paddle.where(
                paddle.cast(loss_mask, 'bool'), acc, paddle.ones_like(acc))
            acc = paddle.sum(paddle.prod(acc, -1))
            return acc

        self.first_step = False

    def validation_step_end(self, log_dict):
        speed = 1. / log_dict['eval_cost']

        if not self.eval_cfgs.cloze_eval:
            self.total_score += log_dict[
                'loss'] * self.configs.Engine.logging_freq / (
                    self.num_tokenized_tokens - 1)
        else:
            self.total_score += log_dict[
                'loss'] * self.configs.Engine.logging_freq

        logger.info("[eval] epoch: %d, batch: %d, %s: %.9f, speed: %.2f step/s"
                    % (log_dict['epoch'], log_dict['batch'], self.score_name,
                       self.total_score, speed))

    def validation_epoch_end(self, log_dict):
        if not self.eval_cfgs.cloze_eval:
            total_loss = float(self.total_score)
            ppl = math.exp(min(20, total_loss))
            token_ratio = (self.num_tokenized_tokens - 1) / (
                self.num_original_tokens - 1)
            adjusted_ppl = math.exp(min(20, total_loss * token_ratio))
            string = ' validation results on {} | '.format(
                self.eval_cfgs.eval_path)
            string += 'avg loss: {:.4E} | '.format(total_loss)
            string += 'ppl: {:.4E} | '.format(ppl)
            string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
            string += 'token ratio: {} |'.format(token_ratio)
        else:
            num_correct = float(self.total_score)
            acc = float(num_correct / self.num_examples)
            string = ' validation results on {} | '.format(
                self.eval_cfgs.eval_path)
            string += 'number correct: {:.4E} | '.format(num_correct)
            string += 'total examples: {:.4E} | '.format(self.num_examples)
            string += 'avg accuracy: {:.4E}'.format(acc)

        logger.info(string)
