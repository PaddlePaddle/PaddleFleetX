#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""LR schedulers."""

import math

import paddle
import paddle.fluid.layers as layers
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter


def linear_warmup_and_linear_decay(learning_rate, end_lr, warmup_steps, max_training_steps):
    """Applies linear warmup and linear decay to the learning rate."""
    dtype = "float32"

    with paddle.static.default_main_program()._lr_schedule_guard():
        lr = paddle.static.create_global_var(
            shape=[1],
            value=0.0,
            dtype=dtype,
            persistable=True,
            name="learning_rate")

        global_step = _decay_step_counter(1)

        with layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                paddle.assign(warmup_lr, lr)
            with switch.case(global_step < max_training_steps):
                frac = (global_step - warmup_steps) / (max_training_steps - warmup_steps)
                decayed_lr = learning_rate + (end_lr - learning_rate) * frac
                paddle.assign(decayed_lr, lr)
            with switch.default():
                learning_rate = paddle.full(
                    shape=[1], dtype=dtype, fill_value=end_lr)
                paddle.assign(learning_rate, lr)
        return lr


def linear_warmup_and_cosine_decay(learning_rate, end_lr, warmup_steps, max_training_steps):
    """Applies linear warmup and cosine decay to the learning rate."""
    dtype = "float32"

    with paddle.static.default_main_program()._lr_schedule_guard():
        lr = paddle.static.create_global_var(
            shape=[1],
            value=0.0,
            dtype=dtype,
            persistable=True,
            name="learning_rate")

        global_step = _decay_step_counter(1)

        with layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                paddle.assign(warmup_lr, lr)
            with switch.case(global_step < max_training_steps):
                frac = 0.5 * (paddle.cos((global_step - warmup_steps) * math.pi / (max_training_steps - warmup_steps)) + 1)
                decayed_lr = end_lr + (learning_rate - end_lr) * frac
                paddle.assign(decayed_lr, lr)
            with switch.default():
                learning_rate = paddle.full(
                    shape=[1], dtype=dtype, fill_value=end_lr)
                paddle.assign(learning_rate, lr)
        return lr


def linear_warmup_and_invsqrt_decay(learning_rate, warmup_steps, decay_steps):
    """Applies linear warmup and invsqrt decay to the learning rate."""
    dtype = "float32"

    with paddle.static.default_main_program()._lr_schedule_guard():
        lr = paddle.static.create_global_var(
            shape=[1],
            value=0.0,
            dtype=dtype,
            persistable=True,
            name="learning_rate")

        global_step = _decay_step_counter(1)

        with layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                paddle.assign(warmup_lr, lr)
            with switch.default():
                decayed_lr = lr * paddle.sqrt(decay_steps / (global_step - warmup_steps + decay_steps))
                paddle.assign(decayed_lr, lr)
        return lr
