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

import logging
import os
import sys
import copy
import argparse
import codecs
import yaml
import numpy as np

import paddle
import paddle.distributed as dist
from paddle.fluid import core
from paddle.fluid.reader import use_pinned_memory

from ppfleetx.distributed.apis import env
from ppfleetx.utils.log import logger, advertise
from ppfleetx.utils import check

__all__ = ['get_config', 'print_config']


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

    def setdefault(self, k, default=None):
        if k not in self or self[k] is None:
            self[k] = default
            return default
        else:
            return self[k]


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
    advertise()
    print_dict(config)


def check_config(config):
    """
    Check config
    """
    # global_batch_size = config.get("")

    global_config = config.get('Global')
    check.check_version()
    device = global_config.get('device', 'gpu')
    device = device.lower()
    if device in ['gpu', 'xpu', 'rocm', 'npu', "cpu"]:
        check.check_device(device)
    else:
        raise ValueError(
            f"device({device}) is not in ['gpu', 'xpu', 'rocm', 'npu', 'cpu'],\n"
            "Please ensure the config option Global.device is one of these devices"
        )


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

    process_dist_config(config)
    process_global_configs(config)
    create_attr_dict(AttrDict(config))

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


def is_fused_matmul_bias_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
        return hasattr(core.eager.ops.legacy, 'fused_gemm_epilogue')
    else:
        return False


def process_dist_config(configs):
    """
    process distributed strategy for hybrid parallel
    """
    nranks = dist.get_world_size()

    config = configs['Distributed']

    config.setdefault("hcg", "HybridCommunicateGroup")
    mp_degree = config.setdefault("mp_degree", 1)
    pp_degree = config.setdefault("pp_degree", 1)
    pp_recompute_interval = config.setdefault("pp_recompute_interval", 1)

    # sharding default
    sharding_config = config['sharding']
    sharding_degree = sharding_config.setdefault("sharding_degree", 1)
    sharding_stage = sharding_config.setdefault('sharding_stage', 2)
    sharding_offload = sharding_config.setdefault('sharding_offload', False)
    reduce_overlap = sharding_config.setdefault('reduce_overlap', False)
    broadcast_overlap = sharding_config.setdefault('broadcast_overlap', False)

    other_degree = mp_degree * pp_degree * sharding_degree

    assert nranks % other_degree == 0, "unreasonable config of dist_strategy."
    dp_degree = config.setdefault("dp_degree", nranks // other_degree)
    assert nranks % dp_degree == 0, "unreasonable config of dist_strategy."
    assert nranks == dp_degree * other_degree, \
        "Mismatched config using {} cards with dp_degree[{}]," \
            "mp_degree[{}], pp_degree[{}] and sharding_degree[{}]".format(nranks, \
                dp_degree, mp_degree, pp_degree, sharding_degree)

    if sharding_config['sharding_degree'] > 1 and reduce_overlap:
        if sharding_config['sharding_stage'] == 3 or sharding_config[
                'sharding_offload']:
            sharding_config['reduce_overlap'] = False
            logger.warning(
                "reduce overlap only valid for sharding stage 2 without offload"
            )

    if sharding_config['sharding_degree'] > 1 and broadcast_overlap:
        if sharding_config['sharding_stage'] == 3 or sharding_config[
                'sharding_offload']:
            sharding_config['broadcast_overlap'] = False
            logger.warning(
                "broadcast overlap only valid for sharding stage 2 without offload"
            )

    if broadcast_overlap and configs['Global']['logging_freq'] == 1:
        logger.warning(
            "Set logging_freq to 1 will disable broadcast_overlap. "
            "If you want to overlap the broadcast, please increase the logging_freq."
        )
        sharding_config['broadcast_overlap'] = False

    if sharding_config['sharding_degree'] > 1:
        if getattr(sharding_config, 'broadcast_overlap', False):
            logger.warning(
                "Enable broadcast overlap for sharding will not use pin memory for dataloader"
            )
            use_pinned_memory(False)

    if 'fuse_sequence_parallel_allreduce' not in config:
        config['fuse_sequence_parallel_allreduce'] = False


def process_global_configs(config):
    """
    process global configs for hybrid parallel
    """
    dp_degree = config['Distributed']['dp_degree']
    pp_degree = config['Distributed']['pp_degree']
    sharding_degree = config['Distributed']['sharding']['sharding_degree']

    config['Global']['enable_partial_send_recv'] = True
    if 'sequence_parallel' in config['Model'] and pp_degree > 1:
        if config['Model']['sequence_parallel']:
            config['Global']['enable_partial_send_recv'] = False
            logger.warning(
                "if config.Distributed.pp_degree > 1 and config.Model.sequence_parallel is True, " \
                "config.Global.enable_partial_send_recv will be set False."
            )

    global_cfg = config['Global']

    # Set environment variable
    flags = global_cfg.get("flags", {})
    paddle.set_flags(flags)
    for k, v in flags.items():
        logger.info("Environment variable {} is set {}.".format(k, v))

    if global_cfg['global_batch_size'] is None and global_cfg[
            'local_batch_size'] is None:
        raise ValueError(
            "global_batch_size or local_batch_size should be set.")
    elif global_cfg['global_batch_size'] is not None and global_cfg[
            'local_batch_size'] is not None:
        assert global_cfg['global_batch_size'] // global_cfg['local_batch_size'] == (dp_degree * sharding_degree), "global_batch_size[{}] should be divided by local_batch_size[{}] "\
            "when dp_degree is [{}] and sharding_degree is [{}]".format(global_cfg['global_batch_size'],
            global_cfg['local_batch_size'], dp_degree, sharding_degree)
    elif global_cfg['global_batch_size'] is not None and global_cfg[
            'local_batch_size'] is None:
        assert global_cfg['global_batch_size'] % (dp_degree * sharding_degree) == 0, \
            "global_batch_size[{}] should be divided by dp_degree[{}] times sharding_degree[{}]"\
            .format(global_cfg['global_batch_size'], dp_degree, sharding_degree)
        global_cfg['local_batch_size'] = global_cfg['global_batch_size'] // (
            dp_degree * sharding_degree)
    else:
        global_cfg['global_batch_size'] = global_cfg[
            'local_batch_size'] * dp_degree * sharding_degree
    assert global_cfg['local_batch_size'] % global_cfg['micro_batch_size'] == 0

    # save_load
    global_cfg['save_load'] = global_cfg.get('save_load', {})
    save_load_cfg = global_cfg.save_load
    save_steps = save_load_cfg.get('save_steps', None)
    save_epoch = save_load_cfg.get('save_epoch', None)
    if save_steps is None or save_steps == -1:
        save_load_cfg[
            'save_steps'] = sys.maxsize if sys.version > '3' else sys.maxint

    if save_epoch is None or save_epoch == -1:
        save_load_cfg['save_epoch'] = 1

    save_load_cfg['output_dir'] = save_load_cfg.get('output_dir', './output')
    save_load_cfg['ckpt_dir'] = save_load_cfg.get('ckpt_dir', None)

    # mix_precision
    global_cfg['mix_precision'] = global_cfg.get('mix_precision', {})
    amp_cfg = global_cfg.mix_precision

    amp_cfg['use_pure_fp16'] = amp_cfg.get('use_pure_fp16', False)
    amp_cfg['scale_loss'] = amp_cfg.get('scale_loss', 32768)
    amp_cfg['custom_black_list'] = amp_cfg.get('custom_black_list', None)
    amp_cfg['custom_white_list'] = amp_cfg.get('custom_white_list', None)

    global_cfg['max_steps'] = global_cfg.get('max_steps', 500000)
    global_cfg['eval_freq'] = global_cfg.get('eval_freq', -1)
    global_cfg['eval_iters'] = global_cfg.get('eval_iters', 0)
    global_cfg['logging_freq'] = global_cfg.get('logging_freq', 1)
    global_cfg['num_train_epochs'] = global_cfg.get('num_train_epochs', 1)
    global_cfg['test_iters'] = global_cfg['eval_iters'] * 10 \
            if global_cfg.get('test_iters', None) is None else global_cfg['test_iters']
    global_cfg[
        'accumulate_steps'] = global_cfg.local_batch_size // global_cfg.micro_batch_size


def process_model_configs(config):
    """
    process model configs for hybrid parallel
    """
    configs = config['Model']
    if configs['ffn_hidden_size'] is None:
        configs['ffn_hidden_size'] = 4 * configs['hidden_size']

    if configs['use_recompute']:
        if not configs['recompute_granularity']:
            configs['recompute_granularity'] = 'full'
        if not configs['no_recompute_layers']:
            configs['no_recompute_layers'] = []
        else:
            assert isinstance(configs['no_recompute_layers'],
                              list), "no_recompute_layers should be a list"
            for i in configs['no_recompute_layers']:
                assert isinstance(
                    i, int
                ), "all values in no_recompute_layers should be an integer"
            assert min(configs['no_recompute_layers']) >= 0, \
                "the min value in no_recompute_layers should >= 0"
            assert max(configs['no_recompute_layers']) < configs['num_layers'], \
                "the max value in no_recompute_layers should < num_layers"
            configs['no_recompute_layers'] = sorted(
                list(set(configs['no_recompute_layers'])))

    if configs['fused_linear'] and not is_fused_matmul_bias_supported():
        configs['fused_linear'] = False
        logging.warning(
            "The flag fused_linear only valid for cuda version higher than 11.6, "
            "but the paddle is compiled with cuda " + paddle.version.cuda())

    pp_degree = config.Distributed.pp_degree

    if pp_degree > 1:
        configs['virtual_pp_degree'] = 1 \
            if configs.get('virtual_pp_degree', None) is None \
            else configs['virtual_pp_degree']
        virtual_pp_degree = configs['virtual_pp_degree']
        num_layers = configs.num_layers

        if not (num_layers % (virtual_pp_degree * pp_degree)) == 0:
            assert virtual_pp_degree == 1, "virtual pp doesn't support uneven layer split."
            logger.warning(
                "The num_layers of the model is not divisible by pp_degree." \
                "Receive num_layers: {}, pp_degree: {}.".format(num_layers, pp_degree))
        else:
            assert (num_layers %
                (virtual_pp_degree * pp_degree)) == 0, \
                "The num_layers of the model should be divisible of pp_degree * virtual_pp_degree." \
                "Receive num_layers: {}, pp_degree: {}, virtual_pp_degree: {}.".format(
                num_layers, pp_degree, virtual_pp_degree)

        if virtual_pp_degree > 1:
            local_batch_size = config.Global.local_batch_size
            micro_batch_size = config.Global.micro_batch_size
            acc_steps = local_batch_size // micro_batch_size
            assert acc_steps % pp_degree == 0, "num of microbatches {} should be divisible of pp_degree {} when " \
                                               "using interleave pipeline".format(acc_steps, pp_degree)

        if virtual_pp_degree > 2:
            logger.warning(
                "Setting virtual_pp_degree > 2 may harm the throughput of the pipeline parallel."
            )
    else:
        if configs.get('virtual_pp_degree', None):
            logger.warning("virtual_pp_degree is unuseful.")


def process_optim_configs(config):
    """
    process optim configs for hybrid parallel
    """
    if 'Optimizer' not in config.keys():
        return

    nranks = dist.get_world_size()
    dp_degree = config['Distributed']['dp_degree']
    sharding_degree = config['Distributed']['sharding']['sharding_degree']
    if config['Optimizer']['tensor_fusion']:
        assert nranks == dp_degree * sharding_degree, \
            "tensor_fusion only support single card train or data/sharding parallel train"


def process_data_configs(config):
    """
    process data configs for hybrid parallel
    """
    if 'Data' not in config.keys():
        return

    cfg_global = config['Global']
    cfg_data = config['Data']

    mode_to_num_samples = {
        "Train":
        cfg_global['global_batch_size'] * config['Global']['max_steps'],
        "Eval": cfg_global['global_batch_size'] *
        (config['Global']['max_steps'] // config['Global']['eval_freq'] + 1) *
        config['Global']['eval_iters'],
        "Test":
        cfg_global['global_batch_size'] * config['Global']['test_iters'],
    }

    for mode in ("Train", "Eval", "Test"):
        if mode in cfg_data.keys():
            cfg_data[mode]['dataset']['num_samples'] = mode_to_num_samples[
                mode]


def process_inference_configs(config):
    """
    process inference configs for hybrid parallel
    """
    if 'Inference' not in config.keys():
        return

    configs = config['Inference']

    if configs['model_dir'] is None:
        configs['model_dir'] = config['Global']['save_load']['output_dir']

    if configs['mp_degree'] is None:
        configs['mp_degree'] = config['Distributed']['mp_degree']


def process_configs(config):
    process_data_configs(config)
    process_model_configs(config)
    process_optim_configs(config)
    process_inference_configs(config)

    return config
