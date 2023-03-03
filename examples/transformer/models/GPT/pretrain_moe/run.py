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

import paddle
from paddle.distributed import fleet
import paddle.distributed as dist
from paddle.static import InputSpec

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.abspath(os.path.join(__dir__, '../../../../../')))

from ppfleetx.distributed.apis import env, strategy, io
from ppfleetx.utils.log import logger
from ppfleetx.utils import device, log
from examples.transformer.utils import config as cfg
from examples.transformer.utils import components as cpn

import impls

if __name__ == "__main__":
    # parse config from yaml
    args = cfg.parse_args()
    config = cfg.get_config(args.config, overrides=args.override, show=True)

    # HACK: use certain device
    paddle.set_device(config.Global.device + ':3')

    # init distributed env
    nranks = dist.get_world_size()
    if nranks > 1:
        env.init_dist_env(config)

    env.set_seed(config.Global.seed)

    cfg.process_configs(config)
    cfg.print_config(config)

    # Note: Only for GPTDataset
    dataset_kwargs = {
        "seed": config.Global.seed,
        "model_type": config.Model.name,
    }
    sampler_kwargs = {"batch_size": config.Global.local_batch_size, }

    # build dataloader for training/eval
    dataset_kwargs.update({"mode": "Train"})
    dataset = cpn.build_dataset(config.Data.Train.dataset, **dataset_kwargs)
    sampler = cpn.build_batch_sampler(config.Data.Train.sampler, dataset,
                                      **sampler_kwargs)
    train_data_loader = cpn.build_dataloader(config.Data.Train.loader, dataset,
                                             sampler)

    dataset_kwargs.update({"mode": "Eval"})
    dataset = cpn.build_dataset(config.Data.Eval.dataset, **dataset_kwargs)
    sampler = cpn.build_batch_sampler(config.Data.Eval.sampler, dataset,
                                      **sampler_kwargs)
    valid_data_loader = cpn.build_dataloader(config.Data.Eval.loader, dataset,
                                             sampler)

    # build GPT model
    model, tokenizer, loss_fn = impls.build_model(config)

    if 'Compress' in config:
        from examples.transformer.utils import qat
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

    config.Optimizer.lr.update({
        'epochs': config.Global.num_train_epochs,
        'step_each_epoch': len(train_data_loader),
        'total_steps': config.Global.max_steps,
    })

    # build lr and optim
    lr_scheduler = cpn.build_lr_scheduler(config.Optimizer.lr)
    optimizer = cpn.build_optimizer(
        config.Optimizer,
        model,
        lr_scheduler,
        multi_precision=config.Global.mix_precision.enable)

    # call fleet wrapper
    if nranks > 1:
        model, optimizer, scaler = strategy.wrap_with_fleet(
            config.Distributed, model, optimizer, scaler)

    # load pretrained checkpoints
    load_recovery = {'step': 0, 'epoch': 0, 'rng_state': -1}
    if config.Global.save_load.ckpt_dir is not None:
        io.load(config.Global.save_load.ckpt_dir, model, optimizer, 'train',
                load_recovery)

    # build profiler
    if config.get('Profiler', {}).get('enable', False):
        profiler = cpn.build_profiler(config.Profiler)
    else:
        profiler = None

    # start training
    train_start = log.get_timestamp()

    if load_recovery['rng_state'] != -1:
        paddle.set_cuda_rng_state(load_recovery['rng_state'])

    for epoch_index in range(load_recovery['epoch'],
                             config.Global.num_train_epochs):
        train_epoch_start = log.get_timestamp()

        # time count
        train_losses = []
        train_step_start = log.get_timestamp()

        # Note(GuoxiaWang): Do not use len(train_data_loader()),
        # it will cause a memory leak.
        total_train_batch = len(train_data_loader)
        total_eval_batch = len(
            valid_data_loader) if valid_data_loader is not None else 0
        for step, batch in enumerate(train_data_loader):
            if epoch_index == load_recovery['epoch']:
                if step <= load_recovery['step']:
                    continue

            model.train()
            fit_kwargs = {
                "model": model,
                "loss_fn": loss_fn,
                "scaler": scaler,
                "optimizer": optimizer,
            }

            def forward_func(batch, model, loss_fn):
                tokens, position_ids, labels, loss_mask = batch

                loss_mask.stop_gradient = True
                labels.stop_gradient = True
                position_ids.stop_gradient = True

                preds = model(tokens, position_ids)
                loss = loss_fn(preds, labels, loss_mask)

                return loss

            loss = impls.fit_impl(config, batch, forward_func, **fit_kwargs)
            train_losses.append(loss)

            # training step log
            if (step + 1) % config.Global.logging_freq == 0:
                train_step_cost = log.get_timestamp() - train_step_start
                numpy_losses = [loss.numpy()[0] for loss in train_losses]

                train_cost = train_step_cost \
                    if step == 0 else train_step_cost / config.Global.logging_freq
                speed = 1. / train_cost
                default_global_tokens_num = config.Global.global_batch_size * \
                    config.Data.Train.dataset.max_seq_len
                ips_total = speed * default_global_tokens_num
                ips = ips_total / env.get_data_world_size()

                logger.info(
                    "[train] epoch: %d, batch: %d, loss: %.9f, avg_batch_cost: %.5f sec, speed: %.2f step/s, " \
                    "ips_total: %.0f tokens/s, ips: %.0f tokens/s, learning rate: %.5e"
                    % (epoch_index, step, sum(numpy_losses) / len(numpy_losses), train_cost, speed, ips_total, ips, optimizer.get_lr()))

                train_step_start = log.get_timestamp()
                train_losses = []

            if lr_scheduler is not None:
                lr_scheduler.step()

            optimizer.clear_grad()

            # start eval
            if step > 0 and config.Global.eval_freq > 0 and step % config.Global.eval_freq == 0:
                eval_losses = []
                eval_step_start = log.get_timestamp()

                for eval_step, batch in enumerate(valid_data_loader):
                    loss = impls.eval_impl(config, batch, model, loss_fn)
                    eval_losses.append(loss)

                    if eval_step >= config.Global.eval_iters - 1:
                        break

                eval_step_cost = log.get_timestamp() - eval_step_start
                eval_loss = sum(eval_losses) / len(eval_losses)
                eval_cost = eval_step_cost / config.Global.logging_freq

                logger.info(
                    "[eval] epoch: %d, batch: %d, loss: %.9f, avg_eval_cost: %.5f sec, speed: %.2f step/s"
                    % (epoch_index, eval_step, eval_loss.numpy()[0], eval_cost,
                       1. / eval_cost))

            if step > 0 and config.Global.save_load.save_steps > 0 and \
                step % config.Global.save_load.save_steps == 0:
                device.synchronize()
                io.save(
                    config.Global.save_load.output_dir,
                    model,
                    optimizer,
                    step=step,
                    epoch=epoch_index,
                    sharding_stage=config.Distributed.sharding.sharding_stage)

            if step >= config.Global.max_steps:
                break

            if profiler:
                profiler.step()

        # training epoch log
        train_epoch_cost = log.get_timestamp() - train_epoch_start
        logger.info("[Training] epoch: %d, total time: %.5f sec" %
                    (epoch_index, train_epoch_cost))

    # training end log
    logger.info(
        "The training process is complete and total cost of time for training is : {}".
        format(
            log.convert_timestamp_to_data(log.get_timestamp() - train_start)))

    if profiler:
        cpn.profiler_done(profiler, config.Profiler)
