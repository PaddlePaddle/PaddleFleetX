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
import json
import math

import paddle
from paddle.distributed import fleet
import paddle.distributed as dist
from paddle.static import InputSpec

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../../../')))

from ppfleetx.distributed.apis import env, strategy, io
from ppfleetx.utils.log import logger
from ppfleetx.utils import device, log
from ppfleetx.models.language_model import gpt
from examples.transformer.utils import qat
from examples.transformer.utils import config as cfg
from examples.transformer.utils import components as cpn
import impls

if __name__ == "__main__":
    # parse config from yaml
    args = cfg.parse_args()
    config = cfg.get_config(args.config, overrides=args.override, show=False)

    paddle.set_device(config.Global.device)

    # init distributed env
    nranks = dist.get_world_size()
    if nranks > 1:
        env.init_dist_env(config)

    env.set_seed(config.Global.seed)

    # process configs
    eval_cfgs = config.Offline_Eval
    config.Data.Eval.pop("sampler", None)
    config.Data.Eval.loader.collate_fn = "gpt_collate_fn"
    config.Data.Eval.loader.batch_size = eval_cfgs.batch_size
    config.Data.Eval.dataset.input_dir = eval_cfgs.eval_path
    config.Data.Eval.dataset.max_seq_len = eval_cfgs.max_seq_len
    config.Global.logging_freq = eval_cfgs.logging_freq

    if not eval_cfgs.cloze_eval:
        config.Data.Eval.dataset.name = "LM_Eval_Dataset"
        config.Data.Eval.dataset.overlapping_eval = eval_cfgs.overlapping_eval
    else:
        config.Data.Eval.dataset.name = "Lambada_Eval_Dataset"

    cfg.print_config(config)

    # build GPT model
    model, tokenizer = impls.build_model(config)

    if 'Compress' in config:
        input_spec = [
            InputSpec(
                shape=[None, None], name="tokens", dtype='int64'), InputSpec(
                    shape=[None, None], name="ids", dtype='int64')
        ]
        model, quanter = qat.compress_model(config, model, input_spec)

    if config.Global.mix_precision.enable:
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=config.Global.mix_precision.scale_loss)
        # Note: Save dtype is the same as model dtype. Also can set save_dtype='float32' when 
        # training with pure fp16 strategy, but will cause the rise of memory.
        model = paddle.amp.decorate(models=model, level='O2')
    else:
        scaler = None

    # load pretrained checkpoints
    load_recovery = {'step': 0, 'epoch': 0, 'rng_state': -1}
    if config.Global.save_load.ckpt_dir is not None:
        io.load(
            config.Global.save_load.ckpt_dir,
            model,
            optimizer=None,
            mode='eval',
            load_recovery=load_recovery)

    # build dataset for eval
    if not eval_cfgs.cloze_eval:
        with open(eval_cfgs.eval_path, "rb") as reader:
            entire_data = reader.read().decode('utf-8')

        num_original_tokens = len(entire_data.strip().split(" "))
        entire_data = impls.wikitext_detokenizer(entire_data)
        tokenized_data = tokenizer.encode(entire_data)
        num_tokenized_tokens = len(tokenized_data)
        print('Original Tokens: %d, Detokenized tokens: %d' %
              (num_original_tokens, num_tokenized_tokens))

        dataset = impls.LM_Eval_Dataset(
            tokens=tokenized_data,
            max_seq_len=eval_cfgs.max_seq_len,
            overlapping_eval=eval_cfgs.overlapping_eval,
            eos_token_id=tokenizer.eos_token_id)
    else:
        tokenized_data = []
        tokenized_label = []

        with open(eval_cfgs.eval_path, 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = impls.get_tokens(tokenizer, text)
                tokenized_data.append(tokens)
                tokenized_label.append(labels)

        dataset = impls.Lambada_Eval_Dataset(
            tokens=tokenized_data,
            labels=tokenized_label,
            max_seq_len=eval_cfgs.max_seq_len,
            eos_token_id=tokenizer.eos_token_id)

        num_examples = len(dataset)

    # build dataloader for eval
    valid_data_loader = cpn.build_dataloader(
        config.Data.Eval.loader, dataset, batch_sampler=None)

    # build profiler
    if config.get('Profiler', {}).get('enable', False):
        profiler = cpn.build_profiler(config.Profiler)
    else:
        profiler = None

    # start eval
    model.eval()
    total_score = 0
    score_name = "loss" if not eval_cfgs.cloze_eval else "number correct"
    eval_start = log.get_timestamp()

    if load_recovery['rng_state'] != -1:
        paddle.set_cuda_rng_state(load_recovery['rng_state'])

    for epoch_index in range(config.Global.num_train_epochs):
        eval_epoch_start = log.get_timestamp()

        eval_step_start = log.get_timestamp()
        eval_losses = []
        total_eval_batch = len(valid_data_loader)

        for eval_step, batch in enumerate(valid_data_loader):
            loss = impls.eval_impl(config, batch, model)
            eval_losses.append(float(loss))

            if eval_step > 0 and eval_step % config.Global.logging_freq == 0:
                eval_step_cost = log.get_timestamp() - eval_step_start
                speed = config.Global.logging_freq / eval_step_cost
                eval_loss = sum(eval_losses) / len(eval_losses)

                if not eval_cfgs.cloze_eval:
                    total_score += eval_loss * config.Global.logging_freq / (
                        num_tokenized_tokens - 1)
                else:
                    total_score += eval_loss * config.Global.logging_freq

                logger.info(
                    "[eval] epoch: %d, batch: %d, %s: %.9f, speed: %.2f step/s"
                    % (epoch_index, eval_step, score_name, total_score, speed))

                eval_step_start = log.get_timestamp()
                eval_losses = []

            if eval_step >= config.Global.max_steps:
                break

        eval_epoch_cost = log.get_timestamp() - eval_epoch_start
        logger.info(
            "[eval] epoch {} : evaluting process is complete and cost {}".
            format(epoch_index, log.convert_timestamp_to_data(
                eval_epoch_cost)))

        string = '[eval] epoch {} : validation results on {} | '.format(
            epoch_index, eval_cfgs.eval_path)

        if not eval_cfgs.cloze_eval:
            total_loss = float(total_score)
            ppl = math.exp(min(20, total_loss))
            token_ratio = (num_tokenized_tokens - 1) / (
                num_original_tokens - 1)
            adjusted_ppl = math.exp(min(20, total_loss * token_ratio))

            string += 'avg loss: {:.4E} | '.format(total_loss)
            string += 'ppl: {:.4E} | '.format(ppl)
            string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
            string += 'token ratio: {} |'.format(token_ratio)
        else:
            num_correct = float(total_score)
            acc = float(num_correct / num_examples)

            string += 'number correct: {:.4E} | '.format(num_correct)
            string += 'total examples: {:.4E} | '.format(num_examples)
            string += 'avg accuracy: {:.4E}'.format(acc)

        logger.info(string)

    # evaluting end log
    logger.info(
        "The evaluting process is complete and total cost of time for evaluting is : {}".
        format(
            log.convert_timestamp_to_data(log.get_timestamp() - eval_start)))

    del valid_data_loader

    if profiler:
        cpn.profiler_done(profiler, config.Profiler)
