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

import paddle
from paddle.distributed.sharding import group_sharded_parallel


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

    model, optimizer, scaler = group_sharded_parallel(
        model, optimizer, level, scaler, group, offload, sync_buffers,
        buffer_max_size, segment_size, sync_comm, dp_group)

    def _redefine_opt_step(model, optim):
        grad_func = model._grad_scale
        for opt in model._sharding_optimizers:
            opt_step = opt.step

            def _opt_step(self):
                if self._reduce_overlap:
                    # Wait for the last reduce task. This wait must before grad scale function.
                    assert self._comm_task is not None
                    self._comm_task.wait()

                # NOTE(haohongxiang): To temporarily resolve the problem of INF caused by primary 
                # sharding strategy. Already finish scaling grads by scaled_loss.backward() in engine. 
                # grad_func()
                opt_step()

            opt.step = MethodType(_opt_step, opt)

    if level == "os_g":
        _redefine_opt_step(model, optimizer)

    return model, optimizer, scaler
