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

from types import MethodType
import logging

import paddle
from paddle.optimizer import Optimizer
from paddle.fluid.framework import in_dygraph_mode
from paddle.distributed.utils.log_utils import get_logger
from paddle.distributed.sharding import group_sharded_parallel
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import GroupShardedScaler
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import GroupShardedStage2
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import GroupShardedOptimizerStage2

logger_ = get_logger(logging.WARNING)


def sharding_wrapper(
        model,
        optimizer,
        level,
        scaler=None,
        group=None,
        offload=False,
        sync_buffers=False,
        buffer_max_size=2**23,
        segment_size=2**20,
        sync_comm=False,
        dp_group=None, ):

    assert level in [
        'os',
        'os_g',
        'p_g_os',
    ], "The level must be os, os_g or p_g_os."

    wrapper_func = group_sharded_parallel if level == "p_g_os" \
        else unscaled_group_sharded_parallel

    model, optimizer, scaler = wrapper_func(
        model, optimizer, level, scaler, group, offload, sync_buffers,
        buffer_max_size, segment_size, sync_comm, dp_group)

    return model, optimizer, scaler


def unscaled_group_sharded_parallel(
        model,
        optimizer,
        level,
        scaler=None,
        group=None,
        offload=False,
        sync_buffers=False,
        buffer_max_size=2**23,
        segment_size=2**20,
        sync_comm=False,
        dp_group=None, ):
    """
    Use unscaled_group_sharded_parallel can perform group shared configuration on the model, optimizer and GradScaler.
    """
    # check optition type
    assert isinstance(
        model,
        paddle.nn.Layer), "The model must be the instance of paddle.nn.Layer."
    assert isinstance(
        optimizer, Optimizer
    ), "The optimizer must be the instance of paddle.optimizer.Optimizer."
    assert level in [
        'os',
        'os_g',
    ], "The level must be os_g or p_g_os."

    assert in_dygraph_mode()

    def check_dtype(param):
        return param.dtype == paddle.float16

    params_fp16 = list(filter(check_dtype, model.parameters()))
    if scaler is None and len(params_fp16) > 0:
        raise ValueError("Please enter the correct scaler.")

    # convert model/optimizer/scaler
    logger_.info("*" * 30)
    logger_.info("Sharded level os uses sharded level os_g achieved now.")
    logger_.info("*" * 30)

    device = paddle.get_device().split(":")[0]

    optimizer = GroupShardedOptimizerStage2(
        params=optimizer._parameter_list,
        optim=optimizer,
        group=group,
        offload=offload,
        dp_group=dp_group,
        device=device)
    model = UnscaledGroupShardedStage2(
        model,
        optimizer,
        group=group,
        sync_buffers=sync_buffers,
        buffer_max_size=buffer_max_size,
        dp_group=dp_group,
        device=device)

    if isinstance(scaler, paddle.amp.GradScaler):
        scaler = GroupShardedScaler(scaler)

    logger_.info("*" * 30)
    logger_.info(
        "If there is a communication hang using group sharded, please check whether the communication operations of each process are unified."
    )
    logger_.info("*" * 30)

    return model, optimizer, scaler


class UnscaledGroupShardedStage2(GroupShardedStage2):
    """
    A wrapper for Sharding Stage2 Layer in Dygraph.
    """

    # NOTE(haohongxiang): To temporarily resolve the problem of INF caused by primary 
    # sharding strategy. Already finish scaling grads by scaled_loss.backward() in engine. 
    # grad_func()
    def _redefine_opt_step(self):
        for opt in self._sharding_optimizers:
            opt_step = opt.step

            def _opt_step(self):
                if self._reduce_overlap:
                    # Wait for the last reduce task. This wait must before grad scale function.
                    assert self._comm_task is not None
                    self._comm_task.wait()

                opt_step()

            opt.step = MethodType(_opt_step, opt)
