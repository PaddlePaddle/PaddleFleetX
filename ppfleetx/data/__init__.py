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

from ppfleetx.data import dataset, sampler, utils
from ppfleetx.utils.log import logger


def build_auto_dataset(config, mode):
    """
    build dataset for auto parallel
    """
    dataset = build_dataset(config, mode)

    collate_fn = None
    if 'collate_fn' in config[mode].keys():
        collate_fn_name = config[mode].pop('collate_fn', None)
        collate_fn = getattr(
            utils, collate_fn_name) if collate_fn_name is not None else None

    dataset.collate_fn = collate_fn
    dataset.sample_split = config[mode].pop('sample_split', None)
    return dataset


def build_dataset(config, mode):
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Dataset mode should be Train, Eval, Test"

    # build dataset
    if mode == 'Eval' and mode not in config:
        return None
    config_dataset = config[mode].dataset
    config_dataset = copy.deepcopy(config_dataset)
    dataset_name = config_dataset.pop('name')
    dataset = eval("dataset.{}".format(dataset_name))(**config_dataset)

    logger.debug("build dataset({}) success...".format(dataset))

    return dataset


def build_dataloader(config, mode):
    dataset = build_dataset(config, mode)

    batch_sampler = None
    # build sampler
    if 'sampler' in config[mode].keys():
        config_sampler = config[mode].sampler
        config_sampler = copy.deepcopy(config_sampler)
        sampler_name = config_sampler.pop("name")
        batch_sampler = eval("sampler.{}".format(sampler_name))(
            dataset, **config_sampler)
        logger.debug("build batch_sampler({}) success...".format(
            batch_sampler))

    collate_fn = None
    config_loader = {}
    # build dataloader
    if 'loader' in config[mode].keys():
        config_loader = config[mode].loader
        config_loader = copy.deepcopy(config_loader)
        collate_fn_name = config_loader.pop('collate_fn', None)
        collate_fn = getattr(
            utils, collate_fn_name) if collate_fn_name is not None else None

    data_loader = paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        **config_loader)

    logger.debug("build data_loader({}) success...".format(data_loader))
    return data_loader
