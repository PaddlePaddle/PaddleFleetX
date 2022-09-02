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

import argparse
import math
import os
import random
import time
import sys
import yaml
import numpy as np

import paddle
sys.path.append("../../")
from examples.imagen.imagen_module import ImagenModule
from fleetx.datasets.imagen import create_imagen_dataloader
from fleetx.core.engine.eager_engine import EagerEngine


def override(dl, ks, v):
    """
    Recursively replace dict of list
    """

    def str2num(v):
        try:
            return eval(v)
        except Exception:
            return v

    assert isinstance(dl, (list, dict)), ("{} should be a list or a dict")
    assert len(ks) > 0, ('lenght of keys should larger than 0')
    if isinstance(dl, list):
        k = str2num(ks[0])
        if len(ks) == 1:
            assert k < len(dl), ('index({}) out of range({})'.format(k, dl))
            dl[k] = str2num(v)
        else:
            override(dl[k], ks[1:], v)
    else:
        if len(ks) == 1:
            # assert ks[0] in dl, ('{} is not exist in {}'.format(ks[0], dl))
            if not ks[0] in dl:
                print('A new filed ({}) detected!'.format(ks[0], dl))
            dl[ks[0]] = str2num(v)
        else:
            override(dl[ks[0]], ks[1:], v)


def override_config(config, options=None):
    """
    Recursively override the config
    """
    if options is not None:
        for opt in options:
            assert isinstance(opt, str), (
                "option({}) should be a str".format(opt))
            assert "=" in opt, (
                "option({}) should contain a ="
                "to distinguish between key and value".format(opt))
            pair = opt.split('=')
            assert len(pair) == 2, ("there can be only a = in the option")
            key, value = pair
            keys = key.split('.')
            override(config, keys, value)
    return config


def parse_yaml():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')

    yaml_args = parser.parse_args()
    yaml_dict = yaml.load(open(yaml_args.config, 'rb'), Loader=yaml.Loader)

    override_config(yaml_dict, yaml_args.override)

    return yaml_dict


def do_train():
    configs = parse_yaml()

    device = paddle.set_device(configs['Global']['device'])

    seed = configs['Global']['seed']
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    module = ImagenModule(configs)

    # TODO(haohongxiang): Only need to send `configs['Engine']` into `EagerEngine`
    engine = EagerEngine(module=module, configs=configs)

    if configs['Engine']['save_load']['ckpt_dir'] is not None:
        engine.load()

    train_data_loader = create_imagen_dataloader(
        configs["DataLoader"], places=device)
    for epoch in range(configs['Engine']['num_train_epochs']):
        engine.fit(train_data_loader=train_data_loader,
                   valid_data_loader=None,
                   epoch=epoch)


if __name__ == "__main__":
    do_train()
