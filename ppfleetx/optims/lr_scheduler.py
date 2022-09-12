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

import math
import numpy
import warnings
from paddle import Tensor
from paddle.optimizer.lr import LRScheduler


class CosineAnnealingWithWarmupDecay(LRScheduler):
    def __init__(self,
                 max_lr,
                 min_lr,
                 warmup_rate,
                 decay_steps,
                 last_epoch=0,
                 verbose=False):

        self.decay_steps = decay_steps
        self.warmup_step = warmup_rate * decay_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(CosineAnnealingWithWarmupDecay, self).__init__(
            max_lr, last_epoch, verbose)

    def get_lr(self):
        if self.warmup_step > 0 and self.last_epoch <= self.warmup_step:
            return float(self.max_lr) * (self.last_epoch) / self.warmup_step

        if self.last_epoch > self.decay_steps:
            return self.min_lr

        num_step_ = self.last_epoch - self.warmup_step
        decay_steps_ = self.decay_steps - self.warmup_step
        decay_ratio = float(num_step_) / float(decay_steps_)
        coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


class ViTLRScheduler(LRScheduler):
    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 decay_type='cosine',
                 linear_end=1e-5,
                 warmup_steps=0,
                 verbose=False,
                 last_epoch=-1,
                 **kwargs):

        self.linear_end = linear_end
        self.T_max = epochs * step_each_epoch
        self.warmup_steps = warmup_steps

        if self.warmup_steps >= self.T_max:
            self.warmup_steps = self.T_max - 1

        self.decay_type = decay_type
        self.last_epoch = last_epoch
        super(ViTLRScheduler, self).__init__(learning_rate, last_epoch,
                                             verbose)

    def get_lr(self):

        progress = (self.last_epoch - self.warmup_steps
                    ) / float(self.T_max - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))

        if self.decay_type == 'linear':
            lr = self.linear_end + (self.base_lr - self.linear_end) * (
                1.0 - progress)
        elif self.decay_type == 'cosine':
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
        if self.warmup_steps:
            lr = lr * min(1.0, self.last_epoch / self.warmup_steps)

        return lr
