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
import random
import sys
import argparse
import yaml
import numpy as np
from collections import defaultdict

sys.path.append("../../")

import paddle
from paddle.distributed import fleet

from fleetx.utils import logger
from fleetx.datasets.vit import build_dataloader
from fleetx.core.engine.eager_engine import EagerEngine

from examples.vit.vit_module import ViTModule


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

    fleet.init(is_collective=True, strategy=None)

    hcg = fleet.get_hybrid_communicate_group()
    dp_rank = hcg.get_data_parallel_rank()

    seed = configs['Global']['seed']

    if seed:
        assert isinstance(seed, int), "The 'seed' must be a integer!"
        seed += dp_rank
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    train_data_loader = build_dataloader(configs["DataLoader"], "Train",
                                         device)
    eval_during_train = False
    if 'Eval' in configs["DataLoader"]:
        eval_during_train = True
        valid_data_loader = build_dataloader(configs["DataLoader"], "Eval",
                                             device)

    epochs = configs['Engine']['num_train_epochs']
    step_each_epoch = len(train_data_loader)
    configs["Optimizer"]["lr"]["step_each_epoch"] = step_each_epoch
    configs["Optimizer"]["lr"]["epochs"] = epochs

    module = ViTModule(configs)
    engine = EagerEngine(module=module, configs=configs)

    output_dir = configs['Engine']['save_load']['output_dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    start_epoch = 0
    if configs['Engine']['save_load']['ckpt_dir'] is not None:
        ckpt_dir = configs['Engine']['save_load']['ckpt_dir']
        model_path = os.path.join(ckpt_dir, "model.pdparams")
        opt_path = os.path.join(ckpt_dir, "model_state.pdopt")
        meta_path = os.path.join(ckpt_dir, "meta_state.pdopt")

        assert os.path.exists(model_path), f"{model_path} is not exists!"
        assert os.path.exists(opt_path), f"{opt_path} is not exists!"
        assert os.path.exists(meta_path), f"{meta_path} is not exists!"

        model_state = paddle.load(model_path)
        engine._module.model.set_state_dict(model_state)

        opt_state = paddle.load(opt_path)
        engine._module.optimizer.set_state_dict(opt_state)

        meta_state = paddle.load(meta_path)
        start_epoch = meta_state['epoch']

    for epoch in range(start_epoch, epochs):
        engine.fit(train_data_loader=train_data_loader, epoch=epoch)
        if eval_during_train:
            engine.evaluate(valid_data_loader=valid_data_loader, epoch=epoch)
            if len(engine._module.acc_list) > 0:
                ret = defaultdict(list)

                for item in engine._module.acc_list:
                    for key, val in item.items():
                        ret[key].append(val)

                msg = ", ".join(
                    [f'{k} = {np.mean(v):.6f}' for k, v in ret.items()])
                logger.info(f"[eval] epoch: {epoch}, {msg}")
                engine._module.acc_list.clear()

        paddle.save(engine._module.model.state_dict(),
                    os.path.join(output_dir, "model.pdparams"))
        paddle.save(engine._module.optimizer.state_dict(),
                    os.path.join(output_dir, "model_state.pdopt"))

        meta_dict = {"epoch": epoch}
        paddle.save(meta_dict, os.path.join(output_dir, "meta_state.pdopt"))
        logger.info(f"Save last model to {output_dir}")


if __name__ == "__main__":
    do_train()
