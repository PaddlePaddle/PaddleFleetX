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
import logging

import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed.fleet import auto
from paddle.optimizer.lr import LRScheduler

from ppfleetx.utils.log import logger
from ppfleetx.core.engine import BasicEngine
from ppfleetx.core.module import BasicModule
from ppfleetx.utils.version import version_check
from ppfleetx.data import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoEngine(BasicEngine):
    def __init__(self, configs, module, optimizer=None, lr=None, mode='train'):
        super().__init__()
        version_check()

        self.mode = mode

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

        if mode == 'train':
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
        self._strategy = self._configs['strategy']

        # save & load
        self._save_steps = self._configs['save_load']['save_steps']
        self._output_dir = self._configs['save_load']['output_dir']
        self._ckpt_dir = self._configs['save_load']['ckpt_dir']

        # engine fit inputs
        collate_fn_name = configs['Data']['collate_fn']
        self.collate_fn = getattr(
            utils, collate_fn_name) if collate_fn_name is not None else None
        self.sample_split = configs['Data']['sample_split']
        self.batch_size = configs['Global']['global_batch_size']

        # init engine
        optimizer = optimizer if mode == 'train' else None
        self._auto_engine = auto.Engine(
            module.model, module.loss_fn, optimizer, strategy=self._strategy)

    def fit(self, epoch=1, train_dataset=None, valid_dataset=None):

        self._auto_engine.fit(train_data=train_dataset,
                              valid_data=valid_dataset,
                              train_sample_split=self.sample_split,
                              valid_sample_split=self.sample_split,
                              epochs=self._num_train_epochs,
                              batch_size=self.batch_size,
                              steps_per_epoch=self._max_steps,
                              valid_steps=self._eval_iters,
                              valid_freq=self._eval_freq,
                              collate_fn=self.collate_fn)

    def evaluate(self, valid_dataset=None):

        self._auto_engine.evaluate(
            valid_data=valid_dataset,
            valid_sample_split=self.sample_split,
            batch_size=self.batch_size,
            steps=self._max_steps,
            collate_fn=self.collate_fn)

    def predict(self, test_dataset=None):

        self._auto_engine.predict(
            test_data=test_dataset,
            test_sample_split=self.sample_split,
            batch_size=self.batch_size,
            steps=self._max_steps,
            collate_fn=self.collate_fn)

    def save(self, training=True):
        if self._output_dir and isinstance(self._output_dir, str):
            path = os.path.join(self._output_dir, "auto")
            self._auto_engine.save(path, training=training)
        else:
            raise TypeError("`save` requires a valid value of `output_dir`.")

    def load(self):
        if self._ckpt_dir and isinstance(self._ckpt_dir, str):
            path = os.path.join(self._ckpt_dir, "auto")
            self._auto_engine.load(path)
        else:
            logger.warning("`load` requires a valid value of `ckpt_dir`.")