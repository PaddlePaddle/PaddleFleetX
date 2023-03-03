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
import numpy as np
import json
import re
import math

import paddle
import paddle.distributed as dist
from ppfleetx.utils.log import logger
from ppfleetx.distributed.apis import env
from ppfleetx.models.language_model import gpt
from ppfleetx.data.tokenizers import GPTTokenizer, GPTChineseTokenizer

MODEL_CLASSES = {
    "GPT": (GPTTokenizer, "gpt2"),
    "GPT-cn": (GPTChineseTokenizer, "gpt-cpm-large-cn"),
}


def build_model(config):
    nranks = dist.get_world_size()
    model_setting = copy.deepcopy(config.Model)

    if 'Compress' in config and 'Quantization' in config.Compress:
        quant_setting = copy.deepcopy(config.Compress.Quantization)
        model_setting['skip_tensor_map'] = quant_setting.get('skip_tensor_map',
                                                             {})
        model_setting['freeze_embedding'] = quant_setting.get(
            'freeze_embedding', False)

    model_name = model_setting.pop("name")
    tokenizer_class, pretrained_name = MODEL_CLASSES[model_name]
    tokenizer = tokenizer_class.from_pretrained(pretrained_name)

    if nranks == 1:
        model = gpt.GPTForPretraining(gpt.GPTModel(**model_setting))
    else:
        raise RuntimeError(
            "Only single-card offline eval is supported in GPTModel now.")

    return model, tokenizer


@paddle.no_grad()
def eval_impl(config, batch, model):
    model.eval()

    use_fp16 = config.Global.mix_precision.enable
    black_list = config.Global.mix_precision.custom_black_list
    white_list = config.Global.mix_precision.custom_white_list

    with paddle.amp.auto_cast(
            use_fp16,
            custom_black_list=black_list,
            custom_white_list=white_list,
            level='O2'):

        tokens, loss_mask, attention_mask, position_ids, labels = batch
        preds = model(tokens, position_ids, attention_mask)

        if not config.Offline_Eval.cloze_eval:
            masked_lm_loss = paddle.nn.functional.cross_entropy(
                preds, labels, reduction="none")
            loss = paddle.sum(masked_lm_loss * loss_mask)

            return loss
        else:
            outputs = paddle.argmax(preds, -1)
            acc = paddle.cast(outputs == labels, 'float32')
            acc = paddle.where(
                paddle.cast(loss_mask, 'bool'), acc, paddle.ones_like(acc))
            acc = paddle.sum(paddle.prod(acc, -1))

            return acc


class LM_Eval_Dataset(paddle.io.Dataset):
    def __init__(self,
                 tokens,
                 max_seq_len,
                 eos_token_id,
                 overlapping_eval=None,
                 **kwargs):
        self.tokens = tokens
        self.seq_len = max_seq_len
        self.pad_idx = eos_token_id
        self.overlapping_eval = overlapping_eval
        if self.overlapping_eval is None:
            self.overlapping_eval = self.seq_len
        self.overlapping_eval = max(1, self.overlapping_eval)

        self.total_targets = len(self.tokens) - 1
        # remove first sequence tokens
        targets = max(self.total_targets - self.overlapping_eval, 0)
        self.total_sequences = max(
            math.ceil(targets / self.overlapping_eval) + 1, 1)

    def __len__(self):
        return self.total_sequences

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]
        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape(
            (1, seq_length, seq_length))

        # the pad and eos tokens do not contribute the loss
        loss_mask = np.ones(seq_length, dtype="float32")
        loss_mask[np.where(np.array(tokens) == self.pad_idx)] = 0.0
        position_ids = np.arange(0, seq_length, dtype="int64")

        # -INF mask value as default
        # attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, loss_mask, attention_mask, position_ids, labels]

    def __getitem__(self, idx):
        start_idx = idx * self.overlapping_eval
        end_idx = start_idx + self.seq_len
        tokens = self.tokens[start_idx:end_idx + 1]
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            tokens += [self.pad_idx] * num_pad
        [tokens, loss_mask, attention_mask, position_ids,
         labels] = self._construct_sample(tokens)
        if self.overlapping_eval != self.seq_len and idx != 0:
            loss_mask[:-self.overlapping_eval] *= 0

        return [tokens, loss_mask, attention_mask, position_ids, labels]


class Lambada_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, tokens, labels, max_seq_len, eos_token_id, **kwargs):
        self.pad_idx = eos_token_id
        self.seq_len = max_seq_len
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]

        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape(
            (1, seq_length, seq_length))

        # the pad and eos tokens do not contribute the loss
        position_ids = np.arange(0, seq_length, dtype="int64")

        # -INF mask value as default
        #attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, attention_mask, position_ids, labels]

    def __getitem__(self, idx):
        tokens = self.tokens[idx][:self.seq_len]
        labels = self.labels[idx]
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            tokens += [self.pad_idx] * num_pad
        loss_mask = np.zeros(self.seq_len, dtype="float32")
        loss_mask[num_tokens - len(labels) - 1:num_tokens - 1] = 1.
        [tokens, attention_mask, position_ids,
         labels] = self._construct_sample(tokens)
        return [tokens, loss_mask, attention_mask, position_ids, labels]


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)

    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")

    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")

    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)

    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def get_tokens(tokenizer, text, strict=True):
    if not strict:
        tokens = tokenizer.encode(text)
        return tokens[:-1], [tokens[-1]]
    last_token = text.split()[-1]
    start_idx = text.rfind(last_token)
    beginning_tokens = tokenizer.encode(text[:start_idx].strip())
    last_token = tokenizer.encode(' ' + last_token)
    return beginning_tokens, last_token
