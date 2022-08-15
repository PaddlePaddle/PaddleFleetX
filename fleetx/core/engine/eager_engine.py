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
import paddle.distributed.fleet as fleet
from paddle.optimizer.lr import LRScheduler

sys.path.append("../../../")
from fleetx.utils import logger
from fleetx.core.engine.basic_engine import BasicEngine
from fleetx.core.module.basic_module import BasicModule


class EagerEngine(BasicEngine):
    """
    """

    def __init__(self, module, configs=None):
        super().__init__()

        if not isinstance(module, BasicModule):
            raise TypeError(
                "'module' must be sub classes of `FleetxModule`, but got: {model.__class__.__name__}."
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

        self._configs = configs

        # configs
        for k, v in configs.items():
            self.__dict__.update({'_{}'.format(k): v})

        if self._use_pure_fp16:
            self._scaler = paddle.amp.GradScaler(
                init_loss_scaling=self._scale_loss)
            self._module.model = paddle.amp.decorate(
                models=self._module.model, level='O2', save_dtype='float32')
        else:
            self._scaler = None

        optimizers = module.configure_optimizers()

        if optimizers and isinstance(optimizers, (
                paddle.optimizer.Optimizer, paddle.fluid.optimizer.Optimizer)):
            self._module.optimizer = optimizers
            self._module.lr_scheduler = None
        elif optimizers and isinstance(optimizers,
                                       tuple) and len(optimizers) == 2:
            if optimizers[0] and not isinstance(
                    optimizers[0], (paddle.optimizer.Optimizer,
                                    paddle.fluid.optimizer.Optimizer)):
                raise TypeError("'optimizer' must be object of class `paddle.optimizer.Optimizer`" \
                            " or `paddle.fluid.optimizer.Optimizer`, but got: {optimizers[0].__class__.__name__}.")
            self._module.optimizer = optimizers[0]

            if optimizers[1] and not isinstance(optimizers[1], (LRScheduler)):
                raise TypeError("'lr_scheduler' must be object of class `paddle.optimizer.lr.LRScheduler`" \
                            ", but got: {optimizers[1].__class__.__name__}.")
            self._module.lr_scheduler = optimizers[1]
        else:
            raise TypeError(
                "Only support optimizer or/and lr_scheduler as outputs of `configure_optimizers`."
            )

        self._module.global_step = 0

    def fit(self, epoch=1, train_data_loader=None, valid_data_loader=None):
        self._module.model.train()

        # time count
        reader_cost = 0.0
        train_cost = 0.0
        reader_start = time.time()

        for step, batch in enumerate(train_data_loader()):
            reader_cost += time.time() - reader_start
            train_start = time.time()

            self._module.global_step += 1
            loss = self._fit_impl(batch)

            # Sync for profile time, delete it may be a little faster
            paddle.device.cuda.synchronize()
            train_cost += time.time() - train_start

            if self._module.global_step % self._logging_freq == 0:
                self._module.training_step_end(loss, epoch, step, reader_cost,
                                               train_cost)

                reader_cost = 0.0
                train_cost = 0.0

            if self._module.global_step % self._eval_freq == 0:
                self._module.model.eval()

                eval_losses = []
                eval_start = time.time()

                for eval_step, batch in enumerate(valid_data_loader):
                    loss = self._evaluate_impl(batch)
                    eval_losses.append(loss)

                    if eval_step >= self._eval_iters - 1:
                        break

                paddle.device.cuda.synchronize()
                eval_cost = time.time() - eval_start
                eval_loss = sum(eval_losses) / len(eval_losses)
                self._module.validation_step_end(eval_loss, epoch, eval_step,
                                                 eval_cost)

                self._module.model.train()

            if self._module.global_step % self._save_steps == 0 or self._module.global_step >= self._max_steps:
                self.save(self._output_dir)

            if self._module.global_step >= self._max_steps:
                logger.info("The training process is complete.")
                del train_data_loader
                return

            reader_start = time.time()

    def _fit_impl(self, batch):
        with paddle.amp.auto_cast(
                self._use_pure_fp16,
                custom_black_list=self._configs.custom_black_list,
                custom_white_list=self._configs.custom_white_list,
                level='O2'):
            loss = self._module.training_step(batch)

        loss_bw = self._scaler.scale(loss) if self._use_pure_fp16 else loss
        self._module.backward(loss_bw)

        if self._use_pure_fp16:
            self._scaler.minimize(self._module.optimizer, loss)
        else:
            self._module.optimizer.step()
        self._module.lr_scheduler.step()
        self._module.optimizer.clear_grad()

        return loss

    @paddle.no_grad()
    def evaluate(self, epoch=1, valid_data_loader=None):
        self._module.model.eval()

        eval_start = time.time()
        for eval_step, batch in enumerate(valid_data_loader):
            self._module.global_step += 1
            loss = self._evaluate_impl(batch)

            paddle.device.cuda.synchronize()
            eval_cost += time.time() - eval_start

            if self._module.global_step % self._logging_freq == 0:
                self._module.validation_step_end(loss, epoch, eval_step,
                                                 eval_cost)
                eval_start = time.time()

            if self._module.global_step >= self._max_steps:
                logger.info("The evaluting process is complete.")
                del valid_data_loader
                return

    @paddle.no_grad()
    def _evaluate_impl(self, batch):
        loss = self._module.validation_step(batch)
        return loss

    @paddle.no_grad()
    def predict(self, epoch=1, test_data_loader=None):
        self._module.model.eval()

        test_start = time.time()
        for test_step, batch in enumerate(test_data_loader):
            self._module.global_step += 1
            loss = self._predict_impl(batch)

            paddle.device.cuda.synchronize()
            test_cost += time.time() - test_start

            if self._module.global_step % self._logging_freq == 0:
                self._module.test_step_end(loss, epoch, test_step, test_cost)
                test_start = time.time()

            if self._module.global_step >= self._max_steps:
                logger.info("The predicting process is complete.")
                del test_data_loader
                return

    @paddle.no_grad()
    def _predict_impl(self, batch):
        loss = self._module.test_step(batch)
        return loss

    def save(self, output_dir):
        if output_dir and isinstance(output_dir, str):
            output_dir = os.path.join(output_dir,
                                      "model_%d" % self._module.global_step)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            logger.info("Save model to %s" % output_dir)
            paddle.save(self._module.model.state_dict(),
                        os.path.join(output_dir, "model.pdparams"))
            paddle.save(self._module.optimizer.state_dict(),
                        os.path.join(output_dir, "model_state.pdopt"))
        else:
            raise TypeError("`save` requires a valid value of `output_dir`.")

    def load(self, ckpt_dir):
        if ckpt_dir and isinstance(ckpt_dir, str):
            logger.info("Try to load checkpoint from %s " % ckpt_dir)
            model_path = os.path.join(ckpt_dir, "model.pdparams")
            opt_path = os.path.join(ckpt_dir, "model_state.pdopt")
            if os.path.exists(model_path):
                model_dict = paddle.load(model_path)
                self._module.model.set_state_dict(model_dict)
            else:
                logger.warning("No optimizer checkpoint file found in %s." %
                               model_path)

            if os.path.exists(opt_path):
                opt_dict = paddle.load(opt_path)
                self._module.optimizer.set_state_dict(opt_dict)
            else:
                logger.warning("No optimizer checkpoint file found in %s." %
                               opt_path)
        else:
            raise TypeError("`load` requires a valid value of `ckpt_dir`.")
