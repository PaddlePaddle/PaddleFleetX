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
import random
import numpy as np

import paddle
import paddle.distributed as dist
from paddle.optimizer.lr import LRScheduler
from paddle.profiler import SummaryView

from ppfleetx.data import dataset, sampler, utils
from ppfleetx.distributed.apis import env
from ppfleetx.utils.log import logger
from ppfleetx.optims import optimizer, grad_clip, lr_scheduler


def build_dataset(config_dataset, **config_kwargs):
    # build dataset
    if config_dataset is not None:
        config_dataset = copy.deepcopy(config_dataset)
        dataset_name = config_dataset.pop('name')
        config_dataset.update(config_kwargs)
        dataset = eval("dataset.{}".format(dataset_name))(**config_dataset)

        logger.debug("build dataset({}) success...".format(dataset))
    else:
        dataset = None

    return dataset


def build_batch_sampler(config_sampler, dataset, **config_kwargs):
    # build sampler
    if config_sampler is not None:
        config_sampler = copy.deepcopy(config_sampler)
        sampler_name = config_sampler.pop("name")
        config_sampler.update(config_kwargs)
        batch_sampler = eval("sampler.{}".format(sampler_name))(
            dataset, **config_sampler)

        logger.debug("build batch_sampler({}) success...".format(
            batch_sampler))
    else:
        batch_sampler = None

    return batch_sampler


def build_dataloader(config_loader,
                     dataset,
                     batch_sampler=None,
                     **config_kwargs):
    collate_fn = None

    if config_loader is not None:
        config_loader = copy.deepcopy(config_loader)
        config_loader.update(config_kwargs)

        collate_fn_cfg = config_loader.pop('collate_fn', None)
        if isinstance(collate_fn_cfg, str):
            collate_fn = getattr(
                utils, collate_fn_cfg) if collate_fn_cfg is not None else None
        elif isinstance(collate_fn_cfg, dict):
            collate_fn_class_name = collate_fn_cfg.pop("name")
            collate_fn = eval("utils.{}".format(collate_fn_class_name))(
                **collate_fn_cfg)

            logger.debug("build collate_fn({}) success...".format(collate_fn))

    def worker_init_fn(worker_id):
        """ set seed in subproces for dataloader when num_workers > 0"""
        np.random.seed(env.get_dp_seed() + worker_id)
        random.seed(env.get_dp_seed() + worker_id)

    data_loader = paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        **config_loader)

    logger.debug("build data_loader({}) success...".format(data_loader))
    return data_loader


def build_lr_scheduler(lr_config):
    if 'name' in lr_config:
        lr_name = lr_config.pop('name')
        lr = eval("lr_scheduler.{}".format(lr_name))(**lr_config)
        if isinstance(lr, LRScheduler):
            return lr
        else:
            return lr()
    else:
        lr = lr_config.learning_rate

    logger.debug("build lr ({}) success..".format(lr))
    return lr


def build_grad_clip(grad_clip_config):
    if grad_clip_config is not None:
        grad_clip_name = grad_clip_config.pop('name', 'ClipGradByGlobalNorm')
        grad_clip = eval("grad_clip.{}".format(grad_clip_name))(
            **grad_clip_config)
        return grad_clip
    else:
        return None


def build_optimizer(config, model, lr_scheduler=None, multi_precision=False):
    config = copy.deepcopy(config)
    if lr_scheduler is not None:
        config.pop('lr')

    grad_clip_config = config.pop('grad_clip', None)
    grad_clip = build_grad_clip(grad_clip_config)

    optim_name = config.pop('name')
    optim = eval("optimizer.{}".format(optim_name))(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        grad_clip=grad_clip,
        multi_precision=multi_precision,
        **config)

    logger.debug("build optimizer ({}) success..".format(optim))
    return optim


def build_profiler(profiler_config):
    profiler = None

    if profiler_config.get('enable', False):
        scheduler = profiler_config.get('scheduler', None)
        profiler_log = profiler_config.get('profiler_log', './profiler_log')
        record_shapes = profiler_config.get('record_shapes', True)
        profile_memory = profiler_config.get('profile_memory', True)
        profiler = paddle.profiler.Profiler(
            targets=[
                paddle.profiler.ProfilerTarget.CPU,
                paddle.profiler.ProfilerTarget.GPU
            ],
            scheduler=scheduler,
            on_trace_ready=paddle.profiler.export_chrome_tracing(profiler_log),
            record_shapes=record_shapes,
            profile_memory=profile_memory)
        profiler.start()
        logger.warning("Profiler is enabled, do not enable it in production.")

    return profiler


def profiler_done(profiler, profiler_config):
    if not profiler:
        return

    logger.info("Profiler finished, prepare to print summary...")

    profiler.stop()

    _print_summary(profiler, profiler_config)
    profiler_log = profiler_config.get('profiler_log', './profiler_log')
    logger.info(
        "For more information please install visualdl and run it with following command:"
    )
    logger.info(
        "-------------------------------------------------------------------------------"
    )
    logger.info(f"visualdl --host 0.0.0.0 --logdir {profiler_log}")
    logger.info(
        "-------------------------------------------------------------------------------"
    )


def _print_summary(profiler, profiler_config):
    views_dict = {
        SummaryView.DeviceView: 'device',
        SummaryView.OverView: 'overview',
        SummaryView.ModelView: 'model',
        SummaryView.DistributedView: 'dist',
        SummaryView.KernelView: 'kernel',
        SummaryView.OperatorView: 'op',
        SummaryView.MemoryView: 'mem',
        SummaryView.MemoryManipulationView: 'memcpy',
        SummaryView.UDFView: 'udf',
    }

    default_views = [
        SummaryView.OverView,
        SummaryView.ModelView,
        SummaryView.KernelView,
        SummaryView.OperatorView,
    ]

    def gen_views(cfg):
        # print all summary view if detailed=True
        if profiler_config.get('detailed', False):
            return None

        views = []
        # override default view with user defined value if detailed=False
        for view in SummaryView:
            v = profiler_config.get('summary', {}).get(views_dict[view], None)
            if v is True or (v is None and view in default_views):
                views.append(view)

        return views or None

    profiler.summary(
        sorted_by=paddle.profiler.SortedKeys.GPUTotal,
        views=gen_views(profiler_config))
