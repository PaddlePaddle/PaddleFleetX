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
import json
import math
import time
import sys
import random
import numpy as np

import paddle
from paddle.io import DataLoader

sys.path.append("../../../")
from fleetx.models.gpt_model.modeling import GPTModel, GPTForPretraining
from fleetx.data.tokenizers import GPTTokenizer
from fleetx.data.sampler import Stack, Tuple
from fleetx.datasets.lm_test import LM_Eval_Dataset, Lambada_Eval_Dataset, wikitext_detokenizer
from fleetx.utils import logger
from examples.gpt.tools import parse_yaml


def get_tokens(tokenizer, text, strict=True):
    if not strict:
        tokens = tokenizer.encode(text)
        return tokens[:-1], [tokens[-1]]
    last_token = text.split()[-1]
    start_idx = text.rfind(last_token)
    beginning_tokens = tokenizer.encode(text[:start_idx].strip())
    last_token = tokenizer.encode(' ' + last_token)
    return beginning_tokens, last_token


def create_eval_dataset(configs):
    val_dataloader = None
    eval_batch_size = configs['batch_size']
    seq_len = configs['max_seq_len']

    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    if not configs['cloze_eval']:
        with open(configs['eval_path'], "rb") as reader:
            entire_data = reader.read().decode('utf-8')
        num_original_tokens = len(entire_data.strip().split(" "))
        entire_data = wikitext_detokenizer(entire_data)
        tokenized_data = tokenizer.encode(entire_data)
        num_tokenized_tokens = len(tokenized_data)
        print('Original Tokens: %d, Detokenized tokens: %d' %
              (num_tokenized_tokens, num_original_tokens))
        val_dataset = LM_Eval_Dataset(tokenized_data, seq_len,
                                      tokenizer.eos_token_id,
                                      configs['overlapping_eval'])
    else:
        tokenized_data = []
        tokenized_label = []
        with open(configs['eval_path'], 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = get_tokens(tokenizer, text)
                tokenized_data.append(tokens)
                tokenized_label.append(labels)
        val_dataset = Lambada_Eval_Dataset(
            tokenized_data, tokenized_label, seq_len,
            tokenizer.eos_token_id)  #tokenizer.pad_token_id)
        num_tokenized_tokens = 0
        num_original_tokens = 0

    val_dict = {
        'num_examples': len(val_dataset),
        'num_original_tokens': num_original_tokens,
        'num_tokenized_tokens': num_tokenized_tokens,
    }

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        drop_last=False,
        collate_fn=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()))

    return val_dataloader, val_dict


def do_eval():
    configs = parse_yaml()
    paddle.set_device(configs['Global']['device'])

    seed = configs['Global']['seed']
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    model = GPTForPretraining(GPTModel(configs['Model']))

    if configs['Eval']['ckpt_dir'] is not None:
        logger.info("Load model checkpoint from %s" %
                    configs['Eval']['ckpt_dir'])
        model_dict = paddle.load(os.path.join(configs['Eval']['ckpt_dir']))

        for key, value in model_dict.items():
            model_dict[key] = model_dict[key].astype(paddle.float32)

        model.set_state_dict(model_dict)

    eval_data_loader, eval_dict = create_eval_dataset(configs['Eval'])

    model.eval()
    total_score = 0
    tic_eval = time.time()
    score_name = "loss" if not configs['Eval'][
        'cloze_eval'] else "number correct"
    with paddle.no_grad():
        for step, batch in enumerate(eval_data_loader):
            tokens, loss_mask, attention_mask, position_ids, labels = batch
            preds = model(tokens, position_ids, attention_mask)

            if not configs['Eval']['cloze_eval']:
                masked_lm_loss = paddle.nn.functional.cross_entropy(
                    preds, labels, reduction="none")
                loss = paddle.sum(masked_lm_loss * loss_mask)
                total_score += loss.numpy() / (
                    eval_dict['num_tokenized_tokens'] - 1)

            else:
                outputs = paddle.argmax(preds, -1)
                acc = paddle.cast(outputs == labels, 'float32')
                acc = paddle.where(
                    paddle.cast(loss_mask, 'bool'), acc, paddle.ones_like(acc))
                acc = paddle.sum(paddle.prod(acc, -1))
                total_score += acc.numpy()

            if step % configs['Eval']['logging_freq'] == 0:
                logger.info(
                    "[eval] step %d, batch: %d, %s: %f, speed: %.2f step/s" %
                    (step, step, score_name, total_score,
                     configs['Eval']['logging_freq'] /
                     (time.time() - tic_eval)))
                tic_eval = time.time()

    if not configs['Eval']['cloze_eval']:
        total_loss = float(total_score)
        ppl = math.exp(min(20, total_loss))
        token_ratio = (eval_dict['num_tokenized_tokens'] - 1) / (
            eval_dict['num_original_tokens'] - 1)
        adjusted_ppl = math.exp(min(20, total_loss * token_ratio))
        string = ' validation results on {} | '.format(configs['Eval'][
            'eval_path'])
        string += 'avg loss: {:.4E} | '.format(total_loss)
        string += 'ppl: {:.4E} | '.format(ppl)
        string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
        string += 'token ratio: {} |'.format(token_ratio)
    else:
        num_correct = float(total_score)
        acc = float(num_correct / eval_dict['num_examples'])
        string = ' validation results on {} | '.format(configs['Eval'][
            'eval_path'])
        string += 'number correct: {:.4E} | '.format(num_correct)
        string += 'total examples: {:.4E} | '.format(eval_dict['num_examples'])
        string += 'avg accuracy: {:.4E}'.format(acc)

    logger.info(string)


if __name__ == "__main__":
    do_eval()
