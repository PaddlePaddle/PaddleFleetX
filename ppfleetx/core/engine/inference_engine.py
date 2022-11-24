# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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
import numpy as np
from collections.abc import Sequence, Mapping

import paddle
import paddle.distributed.fleet as fleet

# TensorRT precisions
TRT_PRECISIONS = {
    'fp32': paddle.inference.PrecisionType.Float32,
    'fp16': paddle.inference.PrecisionType.Half,
    'int8': paddle.inference.PrecisionType.Int8,
}


class _StaticGuard(object):
    def __init__(self):
        pass

    def __enter__(self):
        paddle.enable_static()

    def __exit__(self, exc_type, exc_val, exc_tb):
        paddle.disable_static()


class TensorRTConfig(object):
    """
    TensorRT Inference Configuration

    Args:
        max_batch_size (int): The maxmum batch size of input data. Default 1
        workspace_size (int): The size of TensorRT workspace in bytes. Default 1<<30
        min_subgraph_size (int): The minimum subgraph node size to convert subgraph to TensorRT engine. Default 3
        precision (str): The inference precision, can be 'fp32', 'fp16' and 'int8'. Default 'fp16'
        use_static (bool): Whether to serialize and save TensorRT engine. Default False
        use_calib_mode (bool): Whether to use TensorRT calibration. Default False
        collect_shape (bool): Whether to collect dynamic shape. Default False
        shape_range_info_filename (str): Path to dynamic shape range file. Default None
    """

    def __init__(self,
                 max_batch_size=1,
                 workspace_size=1 << 30,
                 min_subgraph_size=3,
                 precision='fp16',
                 use_static=False,
                 use_calib_mode=False,
                 collect_shape=False,
                 shape_range_info_filename=None):
        self.max_batch_size = max_batch_size
        self.workspace_size = eval(workspace_size)
        self.min_subgraph_size = min_subgraph_size
        self.precision = precision
        self.use_static = use_static
        self.use_calib_mode = use_calib_mode
        self.shape_range_info_filename = shape_range_info_filename
        self.collect_shape = collect_shape

    @property
    def precision(self):
        return TRT_PRECISIONS[self._precision]

    @precision.setter
    def precision(self, value):
        print("value", value)
        assert value.lower() in ['fp32', 'fp16', 'int8'], \
            "TensorRT precision can only be 'fp32', 'fp16' or 'int8', " \
            "but got {}".format(value.lower())
        self._precision = value.lower()

    @property
    def collect_shape(self):
        return self._collect_shape

    @collect_shape.setter
    def collect_shape(self, value):
        if value:
            assert self.shape_range_info_filename is not None, \
                    "shape_range_info_filename should be set in " \
                    "collect_shape mode"
        else:
            assert self.shape_range_info_filename and \
                    os.path.isfile(self.shape_range_info_filename), \
                    "shape_range_info_filename {} is not a " \
                    "file".format(self.shape_range_info_filename)
        self._collect_shape = value


class InferenceEngine(object):
    """
    Model Parallel Inference Engine

    Args:
        model_dir (string): root directory of inference model
        mp_degree (int): model parallel size
        tensorrt_config (TensorRTConfig): configurations for TensorRT inference
    """

    def __init__(self, model_dir, mp_degree=1, tensorrt_config=None):
        self.model_dir = model_dir
        self.mp_degree = mp_degree
        self.tensorrt_config = tensorrt_config

        if mp_degree == 1:
            self.nranks = 1
            self.rank = 0
        else:
            self.nranks = fleet.worker_num()
            self.rank = fleet.worker_index()

        self._check_model()

        self._static_guard = _StaticGuard()
        with self._static_guard:
            self._init_predictor()

    def _check_model(self):
        self.model_file = f"{self.model_dir}/rank_{self.rank}/model.pdmodel"
        self.param_file = f"{self.model_dir}/rank_{self.rank}/model.pdiparams"
        # self.model_file = './{}/auto_dist{}.pdmodel'.format(self.model_dir, self.rank)
        # self.param_file = './{}/auto_dist{}.pdiparams'.format(self.model_dir, self.rank)

    def _generate_comm_init_config(self, rank, nranks):
        ring_id_to_ranks = ','.join(['0'] + [str(i) for i in range(nranks)])
        rank_to_ring_ids = ''.join(['{},0\n'.format(i) for i in range(nranks)])
        comm_str = '[ring_id -> ranks]\n' + ring_id_to_ranks + \
                    '\n[rank -> ring_ids]\n' + rank_to_ring_ids

        config_fname = "/tmp/.comm_config{}.csv".format(rank)
        if os.path.exists(config_fname):
            os.remove(config_fname)
        with open(config_fname, 'w') as f:
            f.write(comm_str)

        return config_fname

    def _init_predictor(self):
        device_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        config = paddle.inference.Config(self.model_file, self.param_file)

        config.enable_memory_optim()
        config.switch_ir_optim(True)
        config.switch_ir_debug()
        config.enable_use_gpu(100, device_id)
        all_pass = [
            "multihead_matmul_fuse_pass_v2",
            "fc_elementwise_layernorm_fuse_pass",
            "embedding_eltwise_layernorm_fuse_pass",
            "constant_folding_pass",
        ]
        for pass_item in all_pass:
            config.delete_pass(pass_item)

        # distributed config
        if self.mp_degree > 1:
            trainer_endpoints = fleet.worker_endpoints()
            current_endpoint = trainer_endpoints[self.rank]

            dist_config = config.dist_config()
            dist_config.set_ranks(self.nranks, self.rank)
            dist_config.set_endpoints(trainer_endpoints, current_endpoint)
            dist_config.enable_dist_model(True)

            config_fname = self._generate_comm_init_config(self.rank,
                                                           self.nranks)
            dist_config.set_comm_init_config(config_fname)
            config.set_dist_config(dist_config)

# TensorRT config
        if self.tensorrt_config:
            config.enable_tensorrt_engine(
                max_batch_size=self.tensorrt_config.max_batch_size,
                workspace_size=self.tensorrt_config.workspace_size,
                min_subgraph_size=self.tensorrt_config.min_subgraph_size,
                precision_mode=self.tensorrt_config.precision,
                use_static=self.tensorrt_config.use_static,
                use_calib_mode=self.tensorrt_config.use_calib_mode)

            if self.tensorrt_config.collect_shape:
                config.collect_shape_range_info(
                    self.tensorrt_config.shape_range_info_filename)
            else:
                config.enable_tuned_tensorrt_dynamic_shape(
                    self.tensorrt_config.shape_range_info_filename, True)

        self.predictor = paddle.inference.create_predictor(config)

    def input_names(self):
        return self.predictor.get_input_names()

    def output_names(self):
        return self.predictor.get_output_names()

    def predict(self, data):
        # data in dict/list format
        with self._static_guard:
            if isinstance(data, Sequence):
                if len(data) != len(self.input_names()):
                    raise ValueError()
                for d, name in zip(data, self.input_names()):
                    handle = self.predictor.get_input_handle(name)
                    handle.copy_from_cpu(np.array(d))
            elif isinstance(data, Mapping):
                # key check
                for k, v in data.items():
                    handle = self.predictor.get_input_handle(k)
                    handle.copy_from_cpu(np.array(v))
            else:
                raise ValueError()

            self.predictor.run()
            return {name: self.predictor.get_output_handle(name).copy_to_cpu() \
                    for name in self.output_names()}
