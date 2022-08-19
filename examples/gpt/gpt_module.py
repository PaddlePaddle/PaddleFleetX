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

import paddle
from paddle.distributed import fleet

sys.path.append("../../../")
from fleetx.utils import logger
from fleetx.optim import lr_scheduler as lr
from fleetx.core.module.basic_module import BasicModule
from fleetx.utils.tensor_fusion_helper import fused_parameters
from fleetx.data.tokenizers import GPTTokenizer


class GPTModule(BasicModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.nranks = paddle.distributed.get_world_size()

        if self.nranks == 1:
            from fleetx.models.gpt_model.modeling import GPTModel, GPTForPretraining, GPTPretrainingCriterion
            self.model = GPTForPretraining(GPTModel(configs['Model']))
            self.loss_fn = GPTPretrainingCriterion()
        else:
            from fleetx.models.gpt_model.modeling_hybrid import GPTModel, GPTForPretraining, GPTPretrainingCriterion, GPTForPretrainingPipe
            hcg = fleet.get_hybrid_communicate_group()
            configs['Model']['topology'] = hcg.topology()
            if self.configs['Distributed']['pp_degree'] == 1:
                self.model = GPTForPretraining(GPTModel(configs['Model']))
            else:
                self.model = GPTForPretrainingPipe(configs['Model'])
            self.loss_fn = GPTPretrainingCriterion()
            del configs['Model']['topology']

        print('>> total parameters: ', len(self.model.parameters()))

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
        speed = self.configs['Engine']['logging_freq'] / log_dict['train_cost']
        default_global_tokens_num = self.configs['Data']['batch_size']['global_batch_size'] * \
            self.configs['Data']['dataset']['max_seq_len']

        logger.info(
            "[train] global step %d, epoch: %d, batch: %d, loss: %.9f, avg_batch_cost: %.5f sec, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
            % (self.global_step, log_dict['epoch'], log_dict['batch'],
               log_dict['loss'], 1. / speed, speed,
               speed * default_global_tokens_num,
               speed * default_global_tokens_num, self.optimizer.get_lr()))

    def configure_optimizers(self):
        self.decay_fused_tensors, self.all_fused_tensors = None, None

        if self.configs['Fused']['tensor_fusion']:
            self.decay_fused_tensors, self.all_fused_tensors = fused_parameters(
                self.model)

        opt_configs = self.configs['Optimizer']
        warmup_step = opt_configs['lr']['warmup_rate'] * opt_configs['lr'][
            'decay_steps']
        lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
            max_lr=opt_configs['lr']['max_lr'],
            min_lr=opt_configs['lr']['min_lr'],
            warmup_step=warmup_step,
            decay_step=opt_configs['lr']['decay_steps'])

        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=opt_configs[
            'grad_clip']) if opt_configs['grad_clip'] > 0 else None

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        if self.configs['Fused']['tensor_fusion']:
            decay_params = [p.name for p in self.decay_fused_tensors]
        else:
            decay_params = [
                p.name for n, p in self.model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ]

        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler
            if lr_scheduler is not None else opt_configs['lr']['max_lr'],
            beta1=opt_configs['adam_beta1'],
            beta2=opt_configs['adam_beta2'],
            epsilon=opt_configs['adam_epsilon'],
            parameters=self.all_fused_tensors
            if self.configs['Fused']['tensor_fusion'] else
            self.model.parameters(),
            weight_decay=opt_configs['weight_decay'],
            grad_clip=clip,
            apply_decay_param_fun=lambda x: x in decay_params,
            multi_precision=self.configs['Engine']['mix_precision'][
                'use_pure_fp16'])
        return optimizer, lr_scheduler

    def validation_step(self, batch):
        tokens, position_ids, labels, loss_mask = batch
        preds = self(tokens, position_ids)
        preds = paddle.cast(preds, dtype="float32")
        loss = self.loss_fn(preds, labels, loss_mask)
        return loss

    def validation_step_end(self, log_dict):
        speed = self.configs['Engine']['logging_freq'] / log_dict['eval_cost']
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
        speed = self.configs['Engine']['logging_freq'] / log_dict['test_cost']
        logger.info(
            "[test] epoch: %d, batch: %d, loss: %.9f, avg_test_cost: %.5f sec, speed: %.2f step/s"
            % (log_dict['epoch'], log_dict['batch'], log_dict['loss'],
               1. / speed, speed))


class GPTHybridModule(GPTModule):
    def pretreating_batch(self, batch):
        if self.configs['Distributed']['pp_degree'] > 1:
            tokens, position_ids, labels, loss_mask = batch
            data = [(tokens, position_ids), (labels, loss_mask)]
            return data
        else:
            return batch

    def training_step_end(self, loss, epoch, step, reader_cost, train_cost):
        avg_loss = loss.numpy()
        speed = self.configs['Engine']['logging_freq'] / (
            reader_cost + train_cost)
        avg_reader_cost = reader_cost / self.configs['Engine']['logging_freq']
        default_global_tokens_num = self.configs['Data']['batch_size']['global_batch_size'] * \
            self.configs['Data']['dataset']['max_seq_len']

        logger.info(
            "[train] global step %d, epoch: %d, batch: %d, loss: %.9f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
            % (self.global_step, epoch, step, avg_loss, avg_reader_cost,
               1. / speed, speed, speed * default_global_tokens_num,
               speed * default_global_tokens_num / self.nranks,
               self.optimizer.get_lr()))


class GPTGenerationModule(BasicModule):
    def __init__(self, configs):
        super().__init__()
        self.global_configs = configs
        self.configs = configs['Generation']
        self.nranks = paddle.distributed.get_world_size()

        if self.nranks == 1:
            from fleetx.models.gpt_model.modeling import GPTModel, GPTForGeneration
            self.model = GPTForGeneration(GPTModel(configs['Model']))
        else:
            raise NotImplementedError

        self.configs['max_dec_len'] = self.adjust_length_to_model(
            self.configs['max_dec_len'], 512)

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

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

        ids, scores = self.model(
            input_ids=input_ids,
            max_length=self.configs['max_dec_len'],
            min_length=self.configs['min_dec_len'],
            bos_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            decode_strategy=self.configs['decode_strategy'],
            temperature=self.configs['temperature'],
            top_k=self.configs['top_k'],
            top_p=self.configs['top_p'],
            num_beams=self.configs['num_beams'],
            length_penalty=self.configs['length_penalty'],
            early_stopping=self.configs['early_stopping'],
            num_return_sequences=self.configs['num_return_sequences'])

        generated_sequences = []
        for i, generated_ids in enumerate(ids):
            # print("*" * 10 + " GENERATED SEQUENCE {} ".format(i) + "*" * 10)
            generated_ids = generated_ids.numpy().tolist()
            # Decode text
            text = self.tokenizer.convert_ids_to_string(generated_ids)
            # Add the prompt at the beginning of the sequence.
            sequence = input_text[i] + text
            generated_sequences.append(sequence)
            # print(sequence)

        return generated_sequences


class GPTAutoModule(BasicModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.nranks = paddle.distributed.get_world_size()

        from examples.gpt.tools import Mesh
        from fleetx.models.gpt_model.modeling_auto import GPTModel, GPTForPretraining, GPTPretrainingCriterion
        configs['Model']['mesh'] = Mesh(configs)

        self.model = GPTForPretraining(GPTModel(configs['Model']))
        self.loss_fn = GPTPretrainingCriterion()

        del configs['Model']['mesh']
        print('>> total parameters: ', len(self.model.parameters()))

    def forward(self, tokens, ids):
        tokens.stop_gradient = True
        ids.stop_gradient = True

        return self.model(tokens, ids)

    def configure_optimizers(self):

        opt_configs = self.configs['Optimizer']
        warmup_step = opt_configs['lr']['warmup_rate'] * opt_configs['lr'][
            'decay_steps']
        lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
            max_lr=opt_configs['lr']['max_lr'],
            min_lr=opt_configs['lr']['min_lr'],
            warmup_step=warmup_step,
            decay_step=opt_configs['lr']['decay_steps'])

        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=opt_configs[
            'grad_clip']) if opt_configs['grad_clip'] > 0 else None

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler
            if lr_scheduler is not None else opt_configs['lr']['max_lr'],
            beta1=opt_configs['adam_beta1'],
            beta2=opt_configs['adam_beta2'],
            epsilon=opt_configs['adam_epsilon'],
            parameters=self.model.parameters(),
            weight_decay=opt_configs['weight_decay'],
            grad_clip=clip,
            apply_decay_param_fun=lambda x: x in decay_params)

        return optimizer
