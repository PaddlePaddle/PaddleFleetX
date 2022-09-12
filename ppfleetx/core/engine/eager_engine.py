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
import paddle.distributed.fleet as fleet
from paddle.optimizer.lr import LRScheduler
from paddle.distributed.sharding import group_sharded_parallel
from paddle.fluid.dygraph.parallel import sync_params_buffers
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients
from paddle.profiler import SummaryView

sys.path.append("../../../")
from ppfleetx.utils import logger
from ppfleetx.core.engine.basic_engine import BasicEngine
from ppfleetx.core.module.basic_module import BasicModule
# from ppfleetx.inference import InferenceEngine, export_inference_model
from ppfleetx.utils.tensor_fusion_helper import all_reduce_parameters
from ppfleetx.utils.version import version_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EagerEngine(BasicEngine):
    """
    The common engine for all models that support single-card and distributed 
    training, validation and test. Only used in eager dygraph mode.
    """

    def __init__(self, configs, module, optimizer=None, lr=None, mode='train'):
        """
        Initialize an engine depending on the user-defined module and configs.

        Args:

            module(BasicModule): user-defined module. After assigning computations 
                and configurations of model/optimizers/lr Schedulers, engine can 
                support the whole loop of training/validation/test.
            
            configs(dict): the configurations that engine needs for training/validation/test 
                loop. Such as mix precision strategy, save&load and the infos of steps/epoches.
        
        Return:

            An instance of `EagerEngine`.

        Examples::

            class TestModule(BasicModule):

                def __init__(self):
                    super().__init__()
                    self.model = paddle.nn.Linear(28 * 28, 10)
                    self.loss_fn = paddle.nn.MSELoss()

                def forward(self, x):
                    return paddle.relu(self.model(x.reshape(-1)))

                def training_step(self, batch):
                    x, y = batch
                    loss = self.loss_fn(self(x), y)
                    return loss

                def configure_optimizers(self):
                    return paddle.optimizer.Adam(
                        parameters=self.model.parameters(), learning_rate=0.02)

            module = TestModule()
            engine = EagerEngine(module, configs)

        """
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
        self._accumulate_steps = self._configs['accumulate_steps']

        self._use_pure_fp16 = self._configs['mix_precision']['use_pure_fp16']
        if mode == 'export' and self._use_pure_fp16:
            logger.info("NOTE: disable use_pure_fp16 in export mode")
            self._use_pure_fp16 = False

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
            if mode == 'train':
                self._scaler = paddle.amp.GradScaler(
                    init_loss_scaling=self._scale_loss)

            # Save dtype is the same as model dtype. Also can set save_dtype='float32' when 
            # training with pure fp16 strategy, but will cause the rise of memory.
            self._module.model = paddle.amp.decorate(
                models=self._module.model, level='O2')
        else:
            self._scaler = None

        self._optimizer = optimizer if mode == 'train' else None
        self._lr_scheduler = lr if mode == 'train' else None

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
        self._inference_engine = None

        self.profiler = None
        if 'Profiler' in configs and configs.get('Profiler', {}).get('enable',
                                                                     False):
            self.profiler_config = configs['Profiler']

            scheduler = self.profiler_config.get('scheduler', None)
            profiler_log = self.profiler_config.get('profiler_log',
                                                    './profiler_log')
            record_shapes = self.profiler_config.get('record_shapes', True)
            profile_memory = self.profiler_config.get('profile_memory', True)
            self.profiler = paddle.profiler.Profiler(
                targets=[
                    paddle.profiler.ProfilerTarget.CPU,
                    paddle.profiler.ProfilerTarget.GPU
                ],
                scheduler=scheduler,
                on_trace_ready=paddle.profiler.export_chrome_tracing(
                    profiler_log),
                record_shapes=record_shapes,
                profile_memory=profile_memory)
            self.profiler.start()
            logger.warning(
                "Profiler is enabled, do not enable it in production.")

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
        self._module.model, self._optimizer, self._scaler = group_sharded_parallel(
            model=self._module.model,
            optimizer=self._optimizer,
            level=level,
            scaler=self._scaler,
            group=self._sharding_group,
            offload=self._sharding_offload)

    def _wrap_3D_parallel(self):
        self._module.model = fleet.distributed_model(self._module.model)
        self._optimizer = fleet.distributed_optimizer(self._optimizer)
        self._scaler = fleet.distributed_scaler(
            self._scaler) if self._scaler is not None else self._scaler

    def _train_one_epoch(self,
                         epoch_index,
                         train_data_loader=None,
                         valid_data_loader=None):

        # time count
        train_cost = 0.0
        train_start = time.time()
        for step, batch in enumerate(train_data_loader()):
            loss = self._fit_impl(batch)
            # Sync for profile time, delete it may be a little faster
            paddle.device.cuda.synchronize()
            train_cost += time.time() - train_start

            if step % self._logging_freq == 0:
                log_dict = {
                    'epoch': epoch_index,
                    'batch': step,
                    'train_cost': train_cost,
                    'loss': loss.numpy(),
                    'lr': self._optimizer.get_lr()
                }
                self._module.training_step_end(log_dict)

                train_start = time.time()
                train_cost = 0.0

            if step % self._eval_freq == 0:
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

                log_dict = {
                    'loss': eval_loss.numpy(),
                    'epoch': self.epoch_index,
                    'batch': eval_step,
                    'eval_cost': eval_cost,
                }
                self._module.validation_step_end(log_dict)

                self._module.model.train()

            # if self._save_by_step > 0 and (
            #         step % self._save_by_step == 0 or
            #         step >= self._max_steps) and self._dp_rank == 0:
            #     self.save()

            if step >= self._max_steps:
                logger.info("The training process is complete.")
                return

            if self.profiler:
                self.profiler.step()

    def fit(self, epoch=1, train_data_loader=None, valid_data_loader=None):
        """
        Run the full process of training/validation/save loop.

        Args:

            epoch(int): the epoch index.
            
            train_data_loader(DataLoader, None): a collection of :class:`paddle.io.DataLoader`, specifying training samples.

            valid_data_loader(DataLoader, None): a collection of :class:`paddle.io.DataLoader`, specifying validation samples.

        """
        self._module.model.train()

        # time count
        train_cost = 0.0
        train_start = time.time()

        for epoch_index in range(epoch):
            self.epoch_index = epoch_index
            self._train_one_epoch(self.epoch_index, train_data_loader,
                                  valid_data_loader)

            paddle.device.cuda.synchronize()
            train_cost += time.time() - train_start
            log_dict = {
                'epoch': epoch_index,
                'train_cost': train_cost,
            }
            self._module.training_epoch_end(log_dict)

            # if self._save_by_epoch > 0 and self.epoch_index % self._save_by_epoch == 0 and self._dp_rank == 0:
            #     self.save()

        if self.profiler:
            self._profiler_done()

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
                    optimizer=self._optimizer,
                    lr_scheduler=self._lr_scheduler,
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
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            self._optimizer.step()

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        self._optimizer.clear_grad()

    @paddle.no_grad()
    def evaluate(self, epoch=1, valid_data_loader=None):
        """
        run one evaluation epoch over the validation set.

        Args:

            epoch(int): the epoch index.

            valid_data_loader(DataLoader, None): a collection of :class:`paddle.io.DataLoader`, specifying validation samples.

        """
        self._module.model.eval()

        eval_start = time.time()
        for eval_step, batch in enumerate(valid_data_loader):
            loss = self._evaluate_impl(batch)

            paddle.device.cuda.synchronize()
            eval_cost = time.time() - eval_start

            if self._module.global_step % self._logging_freq == 0:
                log_dict = {
                    'loss': loss.numpy(),
                    'epoch': epoch,
                    'batch': eval_step,
                    'eval_cost': eval_cost,
                }
                self._module.validation_step_end(log_dict)
                eval_start = time.time()

            if self._module.global_step >= self._max_steps:
                logger.info("The evaluting process is complete.")
                del valid_data_loader
                return

    @paddle.no_grad()
    def _evaluate_impl(self, batch):
        batch = self._module.pretreating_batch(batch)

        with paddle.amp.auto_cast(
                self._use_pure_fp16,
                custom_black_list=self._custom_black_list,
                custom_white_list=self._custom_white_list,
                level='O2'):
            if self._pp_degree == 1:
                loss = self._module.validation_step(batch)
            else:
                loss = self._module.model.eval_batch(batch, compute_loss=True)

        return loss

    @paddle.no_grad()
    def predict(self, epoch=1, test_data_loader=None):
        """
        run one evaluation epoch over the test set.

        Args:

            epoch(int): the epoch index.
            
            test_data_loader(DataLoader, None): a collection of :class:`paddle.io.DataLoader`, specifying test samples.

        """
        self._module.model.eval()

        test_start = time.time()
        for test_step, batch in enumerate(test_data_loader):
            self._module.global_step += 1
            loss = self._predict_impl(batch)

            paddle.device.cuda.synchronize()
            test_cost = time.time() - test_start

            if self._module.global_step % self._logging_freq == 0:
                log_dict = {
                    'loss': loss.numpy(),
                    'epoch': epoch,
                    'batch': test_step,
                    'test_cost': test_cost,
                }
                self._module.test_step_end(log_dict)
                test_start = time.time()

            if self._module.global_step >= self._max_steps:
                logger.info("The predicting process is complete.")
                del test_data_loader
                return

    @paddle.no_grad()
    def _predict_impl(self, batch):
        batch = self._module.pretreating_batch(batch)

        with paddle.amp.auto_cast(
                self._use_pure_fp16,
                custom_black_list=self._custom_black_list,
                custom_white_list=self._custom_white_list,
                level='O2'):
            if self._pp_degree == 1:
                loss = self._module.test_step(batch)
            else:
                loss = self._module.model.eval_batch(batch, compute_loss=True)

        return loss

    def save(self):
        """
        save the state dicts of model and optimizer into an checkpoint.
        """
        if self._output_dir and isinstance(self._output_dir, str):
            output_dir = os.path.join(self._output_dir,
                                      "step_%d" % self._module.global_step)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            logger.info("Save model to %s" % output_dir)

            save_dir = "{}/mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}".format(
                output_dir, self._mp_rank, self._sharding_rank,
                self._pp_rank) if self._distributed else output_dir

            if self._sharding_stage == 3:
                self._module.model.get_all_parameters(convert2cpu=False)
            paddle.save(self._module.model.state_dict(),
                        os.path.join(save_dir, "model.pdparams"))
            paddle.save(self._optimizer.state_dict(),
                        os.path.join(save_dir, "model_state.pdopt"))

            meta_dict = {"global_step": self._module.global_step}
            paddle.save(meta_dict, os.path.join(save_dir, "meta_state.pdopt"))

        else:
            raise TypeError("`save` requires a valid value of `output_dir`.")

    def load(self):
        """
        load the saved checkpoint file and update the state dicts of model and optimizer.
        """
        if self._ckpt_dir and isinstance(self._ckpt_dir, str):
            logger.info("Try to load checkpoint from %s " % self._ckpt_dir)

            load_dir = "{}/mp_{:0>2d}_sharding_{:0>2d}_pp_{:0>2d}".format(
                self._ckpt_dir, self._mp_rank, self._sharding_rank,
                self._pp_rank) if self._distributed else self._ckpt_dir
            model_path = os.path.join(load_dir, "model.pdparams")
            opt_path = os.path.join(load_dir, "model_state.pdopt")
            meta_path = os.path.join(load_dir, "meta_state.pdopt")

            if os.path.exists(model_path):
                model_dict = paddle.load(model_path)
                self._module.model.set_state_dict(model_dict)
            else:
                raise ValueError("No optimizer checkpoint file found in %s." %
                                 model_path)

            if self.mode == 'train':
                if os.path.exists(opt_path):
                    opt_dict = paddle.load(opt_path)
                    self._optimizer.set_state_dict(opt_dict)
                else:
                    raise ValueError(
                        "No optimizer checkpoint file found in %s." % opt_path)

                if os.path.exists(meta_path):
                    meta_dict = paddle.load(meta_path)
                    resume_step = int(self._ckpt_dir.strip("/").split("_")[-1])

                    if resume_step != meta_dict["global_step"]:
                        raise ValueError(
                            "Warning: resume_step is {}, but the step of checkpoint is {}.".
                            format(resume_step, meta_dict["global_step"]))

                    self._module.global_step = resume_step
                else:
                    raise ValueError("No meta checkpoint file found in %s." %
                                     meta_path)

            logger.info("successfully load checkpoints")
        else:
            logger.warning("`load` requires a valid value of `ckpt_dir`.")
            raise TypeError("`load` requires a valid value of `ckpt_dir`.")

    # def export(self):
    #     self._module.model.eval()
    #     input_spec = self._module.input_spec()

    #     save_dir = os.path.join(self._output_dir,
    #                             "rank_{}".format(self._dp_rank), 'model')
    #     export_inference_model(self._module.model, input_spec, save_dir)

    # def inference(self, data):
    #     if self._inference_engine is None:
    #         self._inference_engine = InferenceEngine(
    #             self._inference_configs['model_dir'],
    #             self._inference_configs['mp_degree'])

    #     return self._inference_engine.predict(data)

    def _print_summary(self):
        views_dict = {
            SummaryView.DeviceView: 'device',
            SummaryView.OverView: 'overview',
            SummaryView.ModelView: 'model',
            SummaryView.DistributedView: 'dist',
            SummaryView.KernelView: 'kernel',
            SummaryView.OperatorView: 'op',
            SummaryView.MemoryView: 'mem',
            SummaryView.MemoryManipulationView: 'memcpy',
            SummaryView.UDFView: 'udf',
        }

        default_views = [
            SummaryView.OverView,
            SummaryView.ModelView,
            SummaryView.KernelView,
            SummaryView.OperatorView,
        ]

        def gen_views(cfg):
            # print all summary view if detailed=True
            if self.profiler_config.get('detailed', False):
                return None

            views = []
            # override default view with user defined value if detailed=False
            for view in SummaryView:
                v = self.profiler_config.get('summary', {}).get(
                    views_dict[view], None)
                if v is True or (v is None and view in default_views):
                    views.append(view)

            return views or None

        self.profiler.summary(
            sorted_by=paddle.profiler.SortedKeys.GPUTotal,
            views=gen_views(self.profiler_config))

    def _profiler_done(self):
        if not self.profiler:
            return

        logger.info("Profiler finished, prepare to print summary...")

        self.profiler.stop()

        self._print_summary()
        profiler_log = self.profiler_config.get('profiler_log',
                                                './profiler_log')
        logger.info(
            "For more information please install visualdl and run it with following command:"
        )
        logger.info(
            "-------------------------------------------------------------------------------"
        )
        logger.info(f"visualdl --host 0.0.0.0 --logdir {profiler_log}")
        logger.info(
            "-------------------------------------------------------------------------------"
        )
