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
import math
import numpy as np
import types

import paddle
from paddle.static import InputSpec
import paddle.distributed.fleet as fleet

from ppfleetx.core.module.basic_module import BasicModule
import ppfleetx.models.language_model.gpt as gpt
from ppfleetx.models.language_model.gpt.dygraph.sequence_parallel_utils import register_sequence_parallel_allreduce_hooks
from ppfleetx.utils import env
from ppfleetx.utils.log import logger
import paddleslim
from .utils import process_configs
from ppfleetx.data.tokenizers import GPTTokenizer
from .metrics import *

# TODO(haohongxiang): to solve the problem of cross-reference
import paddlenlp
from paddlenlp.transformers.gpt.tokenizer import GPTChineseTokenizer

MODEL_CLASSES = {
    "GPT": (GPTTokenizer, "gpt2"),
    "MoE": (GPTTokenizer, "gpt2"),
    "GPT-cn": (GPTChineseTokenizer, "gpt-cpm-large-cn"),
}


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

    def qat_model(self, model):
        quanter = paddleslim.dygraph.quant.QAT(
            config=self.configs.Quantization)
        return quanter.quantize(model)

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
        if configs.Model.sequence_parallel:
            register_sequence_parallel_allreduce_hooks(
                self, configs.Engine.accumulate_steps,
                configs.Distributed.fuse_sequence_parallel_allreduce)

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")

        l = model_setting['num_layers']
        h = model_setting['hidden_size']
        v = model_setting['vocab_size']
        s = self.configs.Data.Train.dataset.max_seq_len
        self.get_model_size(l, h, v, s)

        model_name = model_setting.pop("name")
        tokenizer_class, pretrained_name = MODEL_CLASSES[model_name]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_name)

        moe_configs = model_setting.get('moe_configs', {'expert_mode': False})
        assert not moe_configs[
            'expert_mode'], "Not support expert mode in GPT model!"
        model_setting["moe_configs"] = moe_configs

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
            loss_fn = gpt.GPTPretrainingCriterionHybird(
                sequence_parallel=self.configs.sequence_parallel)
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


class GPTFinetuneModule(BasicModule):
    def __init__(self, configs):
        self.nranks = paddle.distributed.get_world_size()
        self.data_world_size = env.get_data_world_size()
        super(GPTFinetuneModule, self).__init__(configs)

        # self.loss_config will be init in super class by get_model()
        assert self.loss_config is not None
        assert 'train' in self.loss_config
        assert 'eval' in self.loss_config

        train_loss = copy.deepcopy(self.loss_config.train)
        train_loss_cls = train_loss.pop('name')
        self.loss_fn = eval(f'paddle.nn.loss.{train_loss_cls}')(**train_loss)

        eval_loss = copy.deepcopy(self.loss_config.eval)
        eval_loss_cls = eval_loss.pop('name')
        self.eval_loss_fn = eval(f'paddle.nn.loss.{eval_loss_cls}')(
            **eval_loss)

        # self.metric_config will be init in super class by get_model()
        assert self.metric_config is not None
        assert 'eval' in self.metric_config

        if 'train' in self.metric_config:
            train_metric = copy.deepcopy(self.metric_config.train)
            train_metric_cls = train_metric.pop('name')
            self.train_metric = eval(f'{train_metric_cls}')(**train_metric)

        eval_metric = copy.deepcopy(self.metric_config.eval)
        eval_metric_cls = eval_metric.pop('name')
        self.eval_metric = eval(f'{eval_metric_cls}')(**eval_metric)

        self.best_metric = 0.0

    def process_configs(self, configs):
        return configs

    def get_model(self):
        model_setting = copy.deepcopy(self.configs.Model)
        model_setting.pop("module")

        self.metric_config = model_setting.pop("metric", None)
        self.loss_config = model_setting.pop("loss", None)

        pretrained = model_setting.pop("pretrained")
        num_classes = model_setting.pop("num_classes", 2)
        assert pretrained is not None

        l = model_setting['num_layers']
        h = model_setting['hidden_size']
        v = model_setting['vocab_size']
        num_heads = model_setting['num_attention_heads']
        s = self.configs.Data.Train.dataset.max_length
        self.get_model_size(l, h, v, s)

        model_name = model_setting.pop("name")
        tokenizer_class, pretrained_name = MODEL_CLASSES[model_name]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_name)

        if self.nranks == 1:
            model = gpt.GPTForSequenceClassification(
                gpt.GPTModel(**model_setting), num_classes)
        else:
            raise NotImplementedError

        pretrained_path = pretrained + ".pdparams"
        assert os.path.exists(
            pretrained_path), f'{pretrained_path} is not exists!'
        model_dict = paddle.load(pretrained_path)

        # Note(GuoxiaWang): Guess whether to convert fused vs non-fused parameters.
        # 'q_proj' vs 'qkv_proj'
        def is_fused(model_state):
            for key in model_state:
                if 'qkv_proj' in key:
                    return True
            return False

        def split_params(model_state, num_layers):
            for idx in range(num_layers):
                qkv_b = model_state.pop(
                    f'gpt.decoder.layers.{idx}.self_attn.qkv_proj.bias')
                qkv_w = model_state.pop(
                    f'gpt.decoder.layers.{idx}.self_attn.qkv_proj.weight')

                qkv_b = qkv_b.reshape((num_heads, 3, -1))
                qkv_w = qkv_w.reshape((h, num_heads, 3, -1))

                q_w, k_w, v_w = np.split(qkv_w, 3, axis=2)
                q_w = q_w.reshape((h, -1))
                k_w = k_w.reshape((h, -1))
                v_w = v_w.reshape((h, -1))

                q_b, k_b, v_b = np.split(qkv_b, 3, axis=1)
                q_b = q_b.reshape((-1))
                k_b = k_b.reshape((-1))
                v_b = v_b.reshape((-1))

                model_state[
                    f'gpt.decoder.layers.{idx}.self_attn.q_proj.bias'] = q_b
                model_state[
                    f'gpt.decoder.layers.{idx}.self_attn.q_proj.weight'] = q_w

                model_state[
                    f'gpt.decoder.layers.{idx}.self_attn.k_proj.bias'] = k_b
                model_state[
                    f'gpt.decoder.layers.{idx}.self_attn.k_proj.weight'] = k_w

                model_state[
                    f'gpt.decoder.layers.{idx}.self_attn.v_proj.bias'] = v_b
                model_state[
                    f'gpt.decoder.layers.{idx}.self_attn.v_proj.weight'] = v_w

            return model_state

        def fuse_params(model_state, num_layers):
            for idx in range(num_layers):
                q_b = model_state.pop(
                    f'gpt.decoder.layers.{idx}.self_attn.q_proj.bias')
                q_w = model_state.pop(
                    f'gpt.decoder.layers.{idx}.self_attn.q_proj.weight')

                k_b = model_state.pop(
                    f'gpt.decoder.layers.{idx}.self_attn.k_proj.bias')
                k_w = model_state.pop(
                    f'gpt.decoder.layers.{idx}.self_attn.k_proj.weight')

                v_b = model_state.pop(
                    f'gpt.decoder.layers.{idx}.self_attn.v_proj.bias')
                v_w = model_state.pop(
                    f'gpt.decoder.layers.{idx}.self_attn.v_proj.weight')

                q_w = q_w.reshape((h, num_heads, -1))
                k_w = k_w.reshape((h, num_heads, -1))
                v_w = v_w.reshape((h, num_heads, -1))

                qkv_w = np.stack([q_w, k_w, v_w], axis=2)
                qkv_w = qkv_w.reshape((h, -1))

                q_b = q_b.reshape((num_heads, -1))
                k_b = k_b.reshape((num_heads, -1))
                v_b = v_b.reshape((num_heads, -1))
                qkv_b = np.stack([q_b, k_b, v_b], axis=1)
                qkv_b = qkv_b.reshape((-1))

                model_state[
                    f'gpt.decoder.layers.{idx}.self_attn.qkv_proj.weight'] = qkv_w
                model_state[
                    f'gpt.decoder.layers.{idx}.self_attn.qkv_proj.bias'] = qkv_b
            return model_state

        fused = is_fused(model.state_dict())
        load_fused = is_fused(model_dict)

        if fused is True and load_fused is False:
            model_dict = fuse_params(model_dict, l)
        elif fused is False and load_fused is True:
            model_dict = split_params(model_dict, l)

        for name, param in model.state_dict().items():
            if name in model_dict and param.dtype != model_dict[name].dtype:
                model_dict[name] = model_dict[name].cast(param.dtype)

        model.set_state_dict(model_dict)
        logger.info(f'Load pretrained weight from {pretrained_path}')

        return model

    def forward(self, tokens):
        return self.model(tokens)

    def training_step(self, batch):
        input_ids, labels = batch

        input_ids.stop_gradient = True
        labels.stop_gradient = True

        logits = self(input_ids)
        loss = self.loss_fn(logits, labels)

        return loss

    def training_step_end(self, log_dict):
        speed = 1. / log_dict['train_cost']
        default_global_tokens_num = self.configs.Global.global_batch_size * \
            self.configs.Data.Train.dataset.max_length

        logger.info(
            "[train] epoch: [%d/%d], step: [%d/%d], learning rate: %.7f, loss: %.9f, avg_batch_cost: %.5f sec, speed: %.2f step/s, " \
            "ips_total: %.0f tokens/s, ips: %.0f tokens/s"
            % (log_dict['epoch'], log_dict['total_epoch'], log_dict['batch'], log_dict['total_batch'], log_dict['lr'], log_dict['loss'], log_dict['train_cost'], speed,
               speed * default_global_tokens_num, speed * default_global_tokens_num / self.data_world_size))

    def validation_step(self, batch):
        input_ids, labels = batch

        input_ids.stop_gradient = True
        labels.stop_gradient = True

        logits = self(input_ids)
        loss = self.eval_loss_fn(logits, labels)
        correct = self.eval_metric.compute(logits, labels)
        self.eval_metric.update(correct)
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
        loss = self.eval_loss_fn(preds, labels, loss_mask)
        return loss

    def test_step_end(self, log_dict):
        speed = 1. / log_dict['test_cost']
        logger.info(
            "[test] epoch: %d, batch: %d, loss: %.9f, avg_test_cost: %.5f sec, speed: %.2f step/s"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'],
               log_dict['test_cost'], speed))

    def get_model_size(self, l, h, v, s):
        P = 12 * l * h * h * (1 + 13 / (12 * h) + (v + s) / (12 * l * h))
        logger.info('Model Size: {:.2f} B'.format(P / 1000.0 / 1000.0 /
                                                  1000.0))

    def training_epoch_end(self, log_dict):
        logger.info("[Training] epoch: %d, total time: %.5f sec" %
                    (log_dict['epoch'], log_dict['train_cost']))

    def validation_epoch_end(self, log_dict):
        res = self.eval_metric.accumulate()
        self.eval_metric.reset()
        if isinstance(self.eval_metric, AccuracyAndF1):
            msg = "acc: %.5f, precision: %.5f, recall: %.5f, f1: %.5f, acc and f1: %.5f" % (
                res[0], res[1], res[2], res[3], res[4])
            metric = res[4]
        elif isinstance(self.eval_metric, Mcc):
            msg = "mcc: %.5f" % (res[0])
            metric = res[0]
        elif isinstance(self.eval_metric, PearsonAndSpearman):
            msg = "pearson: %.5f, spearman: %.5f, pearson and spearman: %.5f" % (
                res[0], res[1], res[2])
            metric = res[2]
        else:
            msg = "acc: %.5f" % (res)
            metric = res

        if metric > self.best_metric:
            self.best_metric = metric

        logger.info(
            "[Eval] epoch: %d, total time: %.5f sec, %s, best_metric: %.5f" %
            (log_dict['epoch'], log_dict['eval_cost'], msg, self.best_metric))


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

        model_name = model_setting.pop("name")
        tokenizer_class, pretrained_name = MODEL_CLASSES[model_name]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_name)

        if self.nranks == 1:
            model = gpt.GPTForGeneration(
                gpt.GPTModel(**model_setting), self.generation_cfgs)
        else:
            assert self.nranks == self.configs.Distributed.dp_degree, \
                "only support single card and data parallel in generation task."
            model = gpt.GPTForGenerationHybrid(
                gpt.GPTModelHybrid(**model_setting), self.generation_cfgs)

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
        self.configs.Data.Eval.loader.collate_fn = "gpt_collate_fn"
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

        moe_configs = model_setting.get('moe_configs', {'expert_mode': False})
        model_setting["moe_configs"] = moe_configs

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
        if paddle.distributed.get_world_size() == 1:
            return

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
