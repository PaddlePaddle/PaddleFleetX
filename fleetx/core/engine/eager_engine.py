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
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.optimizer.lr import LRScheduler
from paddle.distributed.sharding import group_sharded_parallel
from paddle.fluid.dygraph.parallel import sync_params_buffers
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

sys.path.append("../../../")
from fleetx.utils import logger
from fleetx.core.engine.basic_engine import BasicEngine
from fleetx.core.module.basic_module import BasicModule
from fleetx.inference.inference_engine import InferenceEngine
from fleetx.utils.tensor_fusion_helper import all_reduce_parameters


class EagerEngine(BasicEngine):
    """
    """

    def __init__(self, module, configs=None):
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
        self._accumulate_steps = self._configs['accumulate_steps']

        self._use_pure_fp16 = self._configs['mix_precision']['use_pure_fp16']
        self._scale_loss = self._configs['mix_precision']['scale_loss']
        self._custom_black_list = self._configs['mix_precision'][
            'custom_black_list']
        self._custom_white_list = self._configs['mix_precision'][
            'custom_white_list']

        self._save_steps = self._configs['save_load']['save_steps']
        self._output_dir = self._configs['save_load']['output_dir']
        self._ckpt_dir = self._configs['save_load']['ckpt_dir']

        # TODO(haohongxiang): Remove there extra configs after reconstruct of Fleet API
        self._dist_configs = configs['Distributed']
        self._dp_degree = self._dist_configs['dp_degree']
        self._mp_degree = self._dist_configs['mp_degree']
        self._pp_degree = self._dist_configs['pp_degree']
        self._sharding_stage = self._dist_configs['sharding']['sharding_stage']
        self._sharding_degree = self._dist_configs['sharding'][
            'sharding_degree']
        self._sharding_offload = self._dist_configs['sharding'][
            'sharding_offload']
        self._use_recompute = configs['Model']['use_recompute']

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

        # distributed configs
        self._distributed = dist.is_initialized()

        if self._distributed:
            self._hcg = fleet.get_hybrid_communicate_group()
            self._dp_group = self._hcg.get_data_parallel_group()
            self._sharding_group = self._hcg.get_sharding_parallel_group()

            self._dp_rank = self._hcg.get_data_parallel_rank()
            self._mp_rank = self._hcg.get_model_parallel_rank()
            self._pp_rank = self._hcg.get_stage_id()
            self._sharding_rank = self._hcg.get_sharding_parallel_rank()

            if self._hcg.nranks > 1:
                self._wrap_with_fleet()
        else:
            self._dp_rank = 0

        self._module.global_step = 0

        self._inference_configs = configs['Inference']
        self._inference_init = False
        self._inference_engine = None

    def _wrap_with_fleet(self):
        if self._sharding_stage in [2, 3]:
            assert self._mp_degree == self._pp_degree == 1, "sharding stage2/3 will support hybrid parallel later"
            self._wrap_sharding_2_3()
        else:
            self._wrap_3D_parallel()

    def _wrap_sharding_2_3(self):
        if self._dp_degree > 1:
            sync_params_buffers(
                self._module.model,
                comm_group=self._dp_group,
                src_rank=self._dp_group.ranks[0])

        level = "p_g_os" if self._sharding_stage == 3 else "os_g"
        self._module.model, self._module.optimizer, self._scaler = group_sharded_parallel(
            model=self._module.model,
            optimizer=self._module.optimizer,
            level=level,
            scaler=self._scaler,
            group=self._sharding_group,
            offload=self._sharding_offload)

    def _wrap_3D_parallel(self):
        self._module.model = fleet.distributed_model(self._module.model)
        self._module.optimizer = fleet.distributed_optimizer(
            self._module.optimizer)
        self._scaler = fleet.distributed_scaler(
            self._scaler) if self._scaler is not None else self._scaler

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

            if (self._module.global_step % self._save_steps == 0 or
                    self._module.global_step >= self._max_steps
                ) and self._dp_rank == 0:
                self.save()

            if self._module.global_step >= self._max_steps:
                logger.info("The training process is complete.")
                del train_data_loader
                return

            reader_start = time.time()

    def _fit_impl(self, batch):
        batch = self._module.pretreating_batch(batch)
        if self._pp_degree == 1:
            if self._use_recompute and isinstance(self._module.model,
                                                  paddle.DataParallel):
                with self._module.model.no_sync():
                    loss = self._model_forward_backward(batch)
                if not hasattr(self._module, "all_fused_tensors"
                               ) or self._module.all_fused_tensors is None:
                    fused_allreduce_gradients(
                        list(self._module.model.parameters()), None)
                else:
                    all_reduce_parameters(self._module.all_fused_tensors,
                                          self._dp_group)
            else:
                loss = self._model_forward_backward(batch)
            self._optim_update_params()
        else:
            with paddle.amp.auto_cast(
                    self._use_pure_fp16,
                    custom_black_list=self._custom_black_list,
                    custom_white_list=self._custom_white_list,
                    level='O2'):
                loss = self._module.model.train_batch(
                    batch,
                    optimizer=self._module.optimizer,
                    lr_scheduler=self._module.lr_scheduler,
                    scaler=self._scaler)
        return loss

    def _model_forward_backward(self, batch):
        with paddle.amp.auto_cast(
                self._use_pure_fp16,
                custom_black_list=self._custom_black_list,
                custom_white_list=self._custom_white_list,
                level='O2'):
            loss = self._module.training_step(batch)

        loss_bw = self._scaler.scale(loss) if self._use_pure_fp16 else loss
        self._module.backward(loss_bw)
        return loss

    def _optim_update_params(self):
        if self._sharding_stage in [2, 3] and self._dp_degree > 1:
            fused_allreduce_gradients(self._module.model.parameters(),
                                      self._hcg)
            if self._sharding_stage == 3:
                for p in self._module.model.parameters():
                    if hasattr(p, "bw_storage"):
                        assert p.grad is None, "This case shouldn't happen."
                        p.bw_storage.scale_(1.0 / self._dp_group.nranks)
                        dist.all_reduce(p.bw_storage, group=self._dp_group)

        if self._use_pure_fp16:
            self._scaler.step(self._module.optimizer)
            self._scaler.update()
        else:
            self._module.optimizer.step()

        if self._module.lr_scheduler is not None:
            self._module.lr_scheduler.step()

        self._module.optimizer.clear_grad()

    @paddle.no_grad()
    def evaluate(self, epoch=1, valid_data_loader=None):
        self._module.model.eval()

        eval_start = time.time()
        for eval_step, batch in enumerate(valid_data_loader):
            self._module.global_step += 1
            loss = self._evaluate_impl(batch)

            paddle.device.cuda.synchronize()
            eval_cost = time.time() - eval_start

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
        batch = self._module.pretreating_batch(batch)
        if self._pp_degree == 1:
            loss = self._module.validation_step(batch)
        else:
            loss = self._module.model.eval_batch(batch, compute_loss=True)
        return loss

    @paddle.no_grad()
    def predict(self, epoch=1, test_data_loader=None):
        self._module.model.eval()

        test_start = time.time()
        for test_step, batch in enumerate(test_data_loader):
            self._module.global_step += 1
            loss = self._predict_impl(batch)

            paddle.device.cuda.synchronize()
            test_cost = time.time() - test_start

            if self._module.global_step % self._logging_freq == 0:
                self._module.test_step_end(loss, epoch, test_step, test_cost)
                test_start = time.time()

            if self._module.global_step >= self._max_steps:
                logger.info("The predicting process is complete.")
                del test_data_loader
                return

    @paddle.no_grad()
    def _predict_impl(self, batch):
        batch = self._module.pretreating_batch(batch)
        if self._pp_degree == 1:
            loss = self._module.test_step(batch)
        else:
            loss = self._module.model.eval_batch(batch, compute_loss=True)
        return loss

    def save(self):
        if self._output_dir and isinstance(self._output_dir, str):
            self._output_dir = os.path.join(self._output_dir, "model_%d" %
                                            self._module.global_step)
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
            logger.info("Save model to %s" % self._output_dir)

            save_dir = "{}/mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}".format(
                self._output_dir, self._mp_rank, self._sharding_rank,
                self._pp_rank) if self._distributed else self._output_dir
            paddle.save(self._module.model.state_dict(),
                        os.path.join(save_dir, "model.pdparams"))
            paddle.save(self._module.optimizer.state_dict(),
                        os.path.join(save_dir, "model_state.pdopt"))
        else:
            raise TypeError("`save` requires a valid value of `output_dir`.")

    def load(self):
        if self._ckpt_dir and isinstance(self._ckpt_dir, str):
            logger.info("Try to load checkpoint from %s " % self._ckpt_dir)

            load_dir = "{}/mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}".format(
                self._ckpt_dir, self._mp_rank, self._sharding_rank,
                self._pp_rank) if self._distributed else self._ckpt_dir
            model_path = os.path.join(load_dir, "model.pdparams")
            opt_path = os.path.join(load_dir, "model_state.pdopt")
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
            logger.warning("`load` requires a valid value of `ckpt_dir`.")

    def inference(self, data):
        if not self._inference_init:
            self._inference_engine = InferenceEngine(
                self._inference_configs['model_dir'],
                self._inference_configs['mp_degree'])

        return self._inference_engine.predict(data)
