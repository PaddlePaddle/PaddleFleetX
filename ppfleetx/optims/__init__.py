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

from collections import defaultdict
import sys
import copy

import paddle
from paddle.optimizer.lr import LRScheduler

from .lr_scheduler import *
from .optimizer import *
from .grad_clip import *

from ppfleetx.utils.log import logger


def build_lr_scheduler(lr_config):
    if 'name' in lr_config:
        lr_name = lr_config.pop('name')
        lr = eval(lr_name)(**lr_config)
        if isinstance(lr, LRScheduler):
            return lr
        else:
            return lr()
    else:
        lr = lr_config.learning_rate

    logger.debug("build lr ({}) success..".format(lr))
    return lr


def build_grad_clip(grad_clip_config):
    if grad_clip_config is not None:
        grad_clip_name = grad_clip_config.pop('name', 'ClipGradByGlobalNorm')
        grad_clip = eval(grad_clip_name)(**grad_clip_config)
        return grad_clip
    else:
        return None


def build_optimizer(config, model, lr_scheduler=None):
    config = copy.deepcopy(config)
    if lr_scheduler is not None:
        config.pop('lr')

    grad_clip_config = config.pop('grad_clip', None)
    grad_clip = build_grad_clip(grad_clip_config)

    optim_name = config.pop('name')
    optim = eval(optim_name)(learning_rate=lr_scheduler,
                             parameters=model.parameters(),
                             grad_clip=grad_clip,
                             **config)

    logger.debug("build optimizer ({}) success..".format(optim))
    return optim
