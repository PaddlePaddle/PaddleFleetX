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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../../../')))

from ppfleetx.distributed.apis import env, strategy, io
from ppfleetx.utils.log import logger
from ppfleetx.utils import device, log
from ppfleetx.models.language_model import metrics
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
        raise RuntimeError("Only support single-card finetune for GPT model.")
        env.init_dist_env(config)

    env.set_seed(config.Global.seed)
    cfg.print_config(config)

    # build dataloader for training/eval
    dataset = cpn.build_dataset(config.Data.Train.dataset)
    sampler = cpn.build_batch_sampler(config.Data.Train.sampler, dataset)
    train_data_loader = cpn.build_dataloader(config.Data.Train.loader, dataset,
                                             sampler)

    dataset = cpn.build_dataset(config.Data.Eval.dataset)
    sampler = cpn.build_batch_sampler(config.Data.Eval.sampler, dataset)
    valid_data_loader = cpn.build_dataloader(config.Data.Eval.loader, dataset,
                                             sampler)

    # build GPT model
    model, tokenizer, train_loss_fn, eval_loss_fn = impls.build_model(config)

    if config.Global.mix_precision.enable:
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=config.Global.mix_precision.scale_loss)
        # Note: Save dtype is the same as model dtype. Also can set save_dtype='float32' when 
        # training with pure fp16 strategy, but will cause the rise of memory.
        model = paddle.amp.decorate(models=model, level='O2')
    else:
        scaler = None

    # build metric
    model_setting = copy.deepcopy(config.Model)
    metric_config = model_setting.pop("metric", None)

    assert metric_config is not None and 'eval' in metric_config

    if 'train' in metric_config:
        train_metric = copy.deepcopy(metric_config.train)
        train_metric_cls = train_metric.pop('name')
        train_metric = eval("metrics.{}".format(train_metric_cls))(
            **train_metric)

    eval_metric = copy.deepcopy(metric_config.eval)
    eval_metric_cls = eval_metric.pop('name')
    eval_metric = eval("metrics.{}".format(eval_metric_cls))(**eval_metric)

    best_metric = 0.0

    # build lr and optim
    config.Optimizer.lr.update({
        'epochs': config.Global.num_train_epochs,
        'step_each_epoch': len(train_data_loader),
        'total_steps': config.Global.max_steps,
    })

    if 'multi_precision' in config.Optimizer:
        assert config.Optimizer.pop('multi_precision') \
            == config.Global.mix_precision.enable

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
    assert config.Global.get('run_mode',
                             'epoch') == 'epoch', 'run_mode must be epoch'

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
                "scaler": scaler,
                "optimizer": optimizer,
                "loss_fn": train_loss_fn,
            }

            def forward_func(batch, model, loss_fn):
                input_ids, labels = batch
                input_ids.stop_gradient = True
                labels.stop_gradient = True

                logits = model(input_ids)
                loss = loss_fn(logits, labels)

                return loss

            loss = impls.fit_impl(config, batch, forward_func, **fit_kwargs)
            train_losses.append(loss)

            # training step log
            if (step + 1) % config.Global.logging_freq == 0:
                train_step_cost = log.get_timestamp() - train_step_start
                numpy_losses = [float(loss) for loss in train_losses]

                train_cost = train_step_cost \
                    if step == 0 else train_step_cost / config.Global.logging_freq
                speed = 1. / train_cost
                default_global_tokens_num = config.Global.global_batch_size * \
                    config.Data.Train.dataset.max_length
                ips_total = speed * default_global_tokens_num
                ips = ips_total / env.get_data_world_size()

                logger.info(
                    "[train] epoch: [%d/%d], step: [%d/%d], learning rate: %.7f, loss: %.9f, avg_batch_cost: " \
                    "%.5f sec, speed: %.2f step/s, ips_total: %.0f tokens/s, ips: %.0f tokens/s"
                    % (epoch_index, config.Global.num_train_epochs, step, total_train_batch, optimizer.get_lr(),
                    sum(numpy_losses) / len(numpy_losses), train_cost, speed, ips_total, ips))

                train_step_start = log.get_timestamp()
                train_losses = []

            if lr_scheduler is not None:
                lr_scheduler.step()

            optimizer.clear_grad()

            # save model/optim states in 'step' mode
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

            if profiler:
                profiler.step()

        # training epoch log
        train_epoch_cost = log.get_timestamp() - train_epoch_start
        logger.info("[Training] epoch: %d, total time: %.5f sec" %
                    (epoch_index, train_epoch_cost))

        eval_epoch_start = log.get_timestamp()

        # start eval in 'epoch' mode
        eval_step_start = log.get_timestamp()
        eval_losses = []
        total_eval_batch = len(valid_data_loader)

        for eval_step, batch in enumerate(valid_data_loader):
            loss = impls.eval_impl(config, batch, model, eval_loss_fn,
                                   eval_metric)

            eval_losses.append(float(loss))

            if eval_step % config.Global.logging_freq == 0:
                eval_step_cost = log.get_timestamp() - eval_step_start

                speed = 1. / eval_step_cost
                logger.info(
                    "[eval] epoch: %d, batch: %d, loss: %.9f, avg_eval_cost: %.5f sec, speed: %.2f step/s"
                    % (epoch_index, eval_step, sum(eval_losses) /
                       len(eval_losses), eval_step_cost, speed))

                eval_step_start = log.get_timestamp()
                eval_losses = []

        eval_epoch_cost = log.get_timestamp() - eval_epoch_start

        # eval epoch log
        res = eval_metric.accumulate()
        eval_metric.reset()

        if isinstance(eval_metric, metrics.AccuracyAndF1):
            msg = "acc: %.5f, precision: %.5f, recall: %.5f, f1: %.5f, acc and f1: %.5f" % (
                res[0], res[1], res[2], res[3], res[4])
            metric = res[4]
        elif isinstance(eval_metric, metrics.Mcc):
            msg = "mcc: %.5f" % (res[0])
            metric = res[0]
        elif isinstance(eval_metric, metrics.PearsonAndSpearman):
            msg = "pearson: %.5f, spearman: %.5f, pearson and spearman: %.5f" % (
                res[0], res[1], res[2])
            metric = res[2]
        else:
            msg = "acc: %.5f" % (res)
            metric = res

        if metric > best_metric:
            best_metric = metric

        logger.info(
            "[Eval] epoch: %d, total time: %.5f sec, %s, best_metric: %.5f" %
            (epoch_index, eval_epoch_cost, msg, best_metric))

        # save model/optim states in 'epoch' mode
        if config.Global.save_load.save_epoch > 0 and \
            epoch_index % config.Global.save_load.save_steps == 0:
            device.synchronize()
            io.save(
                config.Global.save_load.output_dir,
                model,
                optimizer,
                step=len(train_data_loader),
                epoch=epoch_index,
                sharding_stage=config.Distributed.sharding.sharding_stage)

    # training end log
    logger.info(
        "The training process is complete and total cost of time for training is : {}".
        format(
            log.convert_timestamp_to_data(log.get_timestamp() - train_start)))

    if profiler:
        cpn.profiler_done(profiler, config.Profiler)
