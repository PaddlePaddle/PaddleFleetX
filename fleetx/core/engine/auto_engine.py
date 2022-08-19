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
import time
import sys

import paddle
import paddle.nn as nn

from paddle.static import InputSpec
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.engine import Engine

sys.path.append("../../../")
from fleetx.utils import logger
from fleetx.data.sampler import Stack, Tuple
from fleetx.core.engine.basic_engine import BasicEngine
from fleetx.core.module.basic_module import BasicModule


class AutoEngine(BasicEngine):
    def __init__(self, module, configs=None, strategy=None):
        super().__init__()

        if not isinstance(module, BasicModule):
            raise TypeError(
                "'module' must be sub classes of `BasicModule`, but got: {model.__class__.__name__}."
            )
        self._module = module

        if module.model and not isinstance(
                module.model, nn.Layer) and not callable(module.model):
            raise TypeError(
                "'model' must be sub classes of `paddle.nn.Layer` or any callable function, but got: {module.model.__class__.__name__}."
            )

        if module.loss_fn and not isinstance(
                module.loss_fn, nn.Layer) and not callable(module.loss_fn):
            raise TypeError(
                "'loss_fn' must be sub classes of `paddle.nn.Layer` or any callable function, but got: {module.loss_fn.__class__.__name__}."
            )

        # engine configs
        self._configs = configs['Engine']

        self._max_steps = self._configs['max_steps']
        self._eval_freq = self._configs['eval_freq']
        self._eval_iters = self._configs['eval_iters']
        self._test_iters = self._configs['test_iters']
        self._logging_freq = self._configs['logging_freq']
        self._num_train_epochs = self._configs['num_train_epochs']

        self._save_steps = self._configs['save_load']['save_steps']
        self._output_dir = self._configs['save_load']['output_dir']
        self._ckpt_dir = self._configs['save_load']['ckpt_dir']

        self._data_configs = configs['Data']

        optimizer = module.configure_optimizers()
        if optimizer and not isinstance(optimizer, (
                paddle.optimizer.Optimizer, paddle.fluid.optimizer.Optimizer)):
            raise TypeError(
                    "'optimizer' must be object of class `paddle.optimizer.Optimizer`" \
                        " or `paddle.fluid.optimizer.Optimizer`."
                )

        inputs_spec, labels_spec = self._gen_data_holder()

        if not strategy:
            strategy = fleet.DistributedStrategy()
            strategy.semi_auto = True

        self._auto_engine = Engine(
            module.model, inputs_spec, labels_spec, strategy=strategy)
        self._auto_engine.prepare(optimizer, module.loss_fn)

    def _gen_data_holder(self):
        gbsz = self._data_configs['batch_size']['global_batch_size']
        max_seq_len = self._data_configs['dataset']['max_seq_len']

        tokens = InputSpec([gbsz, max_seq_len], "int64", "tokens")
        position_ids = InputSpec([gbsz, max_seq_len], "int64", "position_ids")
        labels = InputSpec([gbsz, max_seq_len], "int64", "labels")
        loss_mask = InputSpec([gbsz, max_seq_len], "float32", "loss_mask")

        return [tokens, position_ids], [labels, loss_mask]

    def fit(self, epoch=1, train_dataset=None, valid_dataset=None):

        self._auto_engine.fit(
            train_dataset,
            batch_size=self._data_configs['batch_size']['global_batch_size'],
            collate_fn=Tuple(Stack(), Stack(), Stack(), Stack()),
            use_cache=True)

    def evaluate(self, valid_dataset=None):

        self._auto_engine.evaluate(
            valid_dataset,
            batch_size=self._data_configs['batch_size']['global_batch_size'],
            collate_fn=Tuple(Stack(), Stack(), Stack(), Stack()),
            use_cache=True)

    def predict(self, test_dataset):

        self._auto_engine.predict(
            test_dataset,
            batch_size=self._data_configs['batch_size']['global_batch_size'],
            collate_fn=Tuple(Stack(), Stack(), Stack(), Stack()),
            use_cache=True)

    def save(self, training=False):
        if self._output_dir and isinstance(self._output_dir, str):
            path = os.path.join(self._output_dir, "auto")
            self._auto_engine.save(path, training=training)
        else:
            raise TypeError("`save` requires a valid value of `output_dir`.")

    def load(self, ):
        if self._ckpt_dir and isinstance(self._ckpt_dir, str):
            path = os.path.join(self._ckpt_dir, "auto")
            self._auto_engine.load(path)
        else:
            logger.warning("`load` requires a valid value of `ckpt_dir`.")
