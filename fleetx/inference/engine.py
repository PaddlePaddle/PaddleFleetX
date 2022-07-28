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

__all__ = ['TensorRTConfig', 'InferenceEngine']


class _StaticGuard(object):
    def __init__(self):
        pass

    def __enter__(self):
        paddle.enable_static()

    def __exit__(self, exc_type, exc_val, exc_tb):
        paddle.disable_static()


class TensorRTConfig(object):
    def __init_(self,
                max_batch_size=1,
                min_subgraph_size=3,
                precision='fp16',
                use_static=False,
                use_calib_mode=False,
                collect_shape=False,
                shape_range_info_filename=None):
        self.max_batch_size = max_batch_size
        self.min_subgraph_size = min_subgraph_size
        self.precision = precision
        self.use_static = use_static
        self.use_calib_mode = use_calib_mode

        self.collect_shape = collect_shape
        self.shape_range_info_filename = shape_range_info_filename
        if not collect_shape:
            assert os.path.isfile(self.shape_range_info_filename), \
                    "not in collect_shape mode but shape_range_info_filename not found"

    @property
    def precision(self):
        precision_map = {
            "int8": paddle_infer.PrecisionType.Int8,
            "fp16": paddle_infer.PrecisionType.Half,
            "fp32": paddle_infer.PrecisionType.Float32
        }
        return precision_map[self.precision]

    @precision.setter
    def precision(self, value):
        assert value in ['fp32', 'fp16', 'int8'], \
                "only support tensorrt precision 'fp32', 'fp16', 'int8'"
        self.precision = value

    @property
    def collect_shape(self):
        return self.collect_shape

    @collect_shape.setter
    def collect_shape(self, value):
        if not value:
            assert os.path.isfile(self.shape_range_info_filename), \
                    "not in collect_shape mode but shape_range_info_filename not found"
        self.collect_shape = value

    @property
    def shape_range_info_filename(self):
        return self.shape_range_info_filename

    @shape_range_info_filename.setter
    def shape_range_info_filename(self, value):
        assert os.path.isfile(self.shape_range_info_filename), \
                "shape_range_info_filename not exists"
        self.shape_range_info_filename = value


class InferenceEngine(object):
    def __init__(self, model_dir, mp_size=1, tensorrt_config=None):
        self.model_dir = model_dir
        self.mp_size = mp_size

        self.nranks = fleet.worker_num()
        self.rank = fleet.worker_index()
        self._check_model()

        self.tensorrt_config = tensorrt_config

        self._static_guard = _StaticGuard()
        with self._static_guard:
            self._init_predictor()

    def _check_model(self):
        if not os.path.isdir(self.model_dir):
            raise ValueError('model_dir is not a directory')

        rank_path = os.path.join(self.model_dir, "rank_{}".format(self.rank))
        if not os.path.isdir(rank_path):
            raise ValueError('rank_{} directory not found'.format(i))
        model_files = []
        param_files = []
        for fname in os.listdir(rank_path):
            if os.path.splitext(fname)[1] == '.pdmodel':
                model_files.append(fname)
            if os.path.splitext(fname)[1] == '.pdiparams':
                param_files.append(fname)

        def _check_and_get_file(files, tag):
            if len(files) == 0:
                raise ValueError("no {} file found under {}".format(tag,
                                                                    rank_path))
            elif len(files) > 1:
                raise ValueError("multiple {} file found under {}".format(
                    tag, rank_path))
            else:
                return os.path.join(self.model_dir,
                                    'rank_{}'.format(self.rank), fname)

        self.model_file = _check_and_get_file(model_files, 'pdmodel')
        self.param_file = _check_and_get_file(param_files, 'pdiparams')

    def _generate_comm_init_config(self, rank, nranks):
        ring_id_to_ranks = ','.join(['0'] + [str(i) for i in range(nranks)])
        rank_to_ring_ids = ''.join(['{},0\n'.format(i) for i in range(nranks)])
        comm_str = '[ring_id -> ranks]\n' + ring_id_to_ranks + \
                    '\n[rank -> ring_ids]\n' + rank_to_ring_ids

        config_fname = "./.comm_config{}.csv".format(rank)
        if os.path.exists(config_fname):
            os.remove(config_fname)
        with open(config_fname, 'w') as f:
            f.write(comm_str)

        return config_fname

    def _init_predictor(self):
        device_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        trainer_endpoints = fleet.worker_endpoints()
        current_endpoint = trainer_endpoints[self.rank]

        config = paddle.inference.Config(self.model_file, self.param_file)

        config.enable_use_gpu(100, device_id)
        config.switch_use_feed_fetch_ops(False)

        # distributed config
        dist_config = config.dist_config()
        dist_config.set_ranks(self.nranks, self.rank)
        dist_config.set_endpoints(trainer_endpoints, current_endpoint)
        dist_config.enable_dist_model(True)

        config_fname = self._generate_comm_init_config(self.rank, self.nranks)
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
                    handle.copy_from_cpu(d)
            elif isinstance(data, Mapping):
                # key check
                for k, v in data.items():
                    handle = self.predictor.get_input_handle(k)
                    handle.copy_from_cpu(v)
            else:
                raise ValueError()

            self.predictor.run()
            return {name: self.predictor.get_output_handle(name).copy_to_cpu() \
                    for name in self.output_names()}
