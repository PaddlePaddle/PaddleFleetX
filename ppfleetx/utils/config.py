# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import copy
import argparse
import yaml
import codecs
import sys
from . import logger
from . import check
import paddle.distributed as dist

__all__ = ['get_config', 'print_config']


def process_dist_config(config):
    """
    process distributed strategy for hybrid parallel
    """
    # config = config['Distributed']

    nranks = dist.get_world_size()


    config['mp_degree'] = 1 \
        if config.get('mp_degree', None) is None \
        else config['mp_degree']

    config['pp_degree'] = 1 \
        if config.get('pp_degree', None) is None \
        else config['pp_degree']

    config['sharding']['sharding_degree'] = 1 \
        if config['sharding'].get('sharding_degree', None) is None \
        else config['sharding']['sharding_degree']

    other_degree = config['mp_degree'] * config['pp_degree'] * config[
        'sharding']['sharding_degree']

    assert nranks % other_degree == 0, "unreasonable config of dist_strategy."

    if not config.get('dp_degree', None):
        config['dp_degree'] = nranks // other_degree
    else:
        if config['dp_degree'] * other_degree != nranks:
            logger.warning('Mismatched config using {} cards with dp_degree[{}], ' \
                'mp_degree[{}], pp_degree[{}] and sharding_degree[{}]. So adaptively ' \
                'adjust dp_degree to {}'.format(nranks, config['dp_degree'], config['mp_degree'],
                config['pp_degree'], config['sharding']['sharding_degree'], nranks // other_degree))
    assert nranks % config[
        'dp_degree'] == 0, "unreasonable config of dist_strategy."


def process_engine_config(config):
    """
    process engine
    """
    if config.get('save_load', None):
        save_load_cfg = config.save_load
        save_steps = save_load_cfg.get('save_steps', None)
        save_epoch = save_load_cfg.get('save_epoch', None)
        if save_steps is None or save_steps == -1:
            save_load_cfg[
                'save_steps'] = sys.maxsize if sys.version > '3' else sys.maxint

        if save_epoch is None or save_epoch == -1:
            save_load_cfg['save_epoch'] = 1


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        for k, v in self.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


def create_attr_dict(yaml_config):
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""

    def _update_dic(dic, base_dic):
        '''Update config from dic based base_dic
        '''
        base_dic = base_dic.copy()
        dic = dic.copy()

        if dic.get('_inherited_', True) == False:
            dic.pop('_inherited_')
            return dic

        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = _update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def _parse_from_yaml(path):
        '''Parse a yaml file and build config'''

        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        if '_base_' in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop('_base_')
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = _parse_from_yaml(base_path)
            dic = _update_dic(dic, base_dic)
        return dic

    yaml_dict = _parse_from_yaml(cfg_file)
    yaml_config = AttrDict(yaml_dict)

    create_attr_dict(yaml_config)
    return yaml_config


def print_dict(d, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    placeholder = "-" * 60
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", k))
            print_dict(v, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", k))
            for value in v:
                print_dict(value, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))
        if k.isupper():
            logger.info(placeholder)


def print_config(config):
    """
    visualize configs
    Arguments:
        config: configs
    """
    logger.advertise()
    print_dict(config)


def check_config(config):
    """
    Check config
    """
    # global_batch_size = config.get("")

    global_config = config.get('Global')
    check.check_version()
    use_gpu = global_config.get('device', True)
    if use_gpu:
        check.check_gpu()


def override(dl, ks, v):
    """
    Recursively replace dict of list
    Args:
        dl(dict or list): dict or list to be replaced
        ks(list): list of keys
        v(str): value to be replaced
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
                print('A new field ({}) detected!'.format(ks[0], dl))
            dl[ks[0]] = str2num(v)
        else:
            if ks[0] not in dl.keys():
                dl[ks[0]] = {}
                print("A new Series field ({}) detected!".format(ks[0], dl))
            override(dl[ks[0]], ks[1:], v)


def override_config(config, options=None):
    """
    Recursively override the config
    Args:
        config(dict): dict to be replaced
        options(list): list of pairs(key0.key1.idx.key2=value)
            such as: [
                'topk=2',
                'VALID.transforms.1.ResizeImage.resize_short=300'
            ]
    Returns:
        config(dict): replaced config
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


def get_config(fname, overrides=None, show=False):
    """
    Read config from file
    """
    assert os.path.exists(fname), (
        'config file({}) is not exist'.format(fname))
    config = parse_config(fname)
    override_config(config, overrides)

    process_dist_config(config['Distributed'])
    process_engine_config(config['Engine'])

    if show:
        print_config(config)
    check_config(config)
    return config


def parse_args():
    parser = argparse.ArgumentParser("train script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/config.yaml',
        help='config file path')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    args = parser.parse_args()
    return args
