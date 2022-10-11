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

import paddle
import paddle.nn as nn
# nn.CrossEntropyLoss
import paddle.nn.functional as F

__all__ = [
    'ViTCELoss',
    'CELoss',
]


class CELoss(nn.Layer):
    """
    Softmax Cross entropy loss
    """

    def __init__(self, epsilon=None):
        super().__init__()
        if epsilon is not None:
            assert epsilon >= 0 and epsilon <= 1, "epsilon must be in [0, 1]"
        self.epsilon = epsilon

    def _labelsmoothing(self, target, class_num):
        if len(target.shape) == 1 or target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._labelsmoothing(label, class_num)
            x = -F.log_softmax(x, axis=-1)
            loss = paddle.sum(x * label, axis=-1)
        else:
            if label.shape[-1] == x.shape[-1]:
                loss = paddle.sum(-label * F.log_softmax(x, axis=-1), axis=-1)
            else:
                if label.dtype == paddle.int32:
                    label = paddle.cast(label, 'int64')
                loss = F.cross_entropy(x, label=label, soft_label=False)
        loss = loss.mean()
        return loss


class ViTCELoss(nn.Layer):
    """
    ViT style Sigmoid Cross entropy loss
    """

    def __init__(self, epsilon=None):
        super().__init__()
        if epsilon is not None:
            assert epsilon >= 0 and epsilon <= 1, "epsilon must be in [0, 1]"
        self.epsilon = epsilon

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        class_num = x.shape[-1]
        if len(label.shape) == 1 or label.shape[-1] != class_num:
            label = F.one_hot(label, class_num)
            label = paddle.reshape(label, shape=[-1, class_num])
        if self.epsilon is not None:
            # vit style label smoothing
            with paddle.no_grad():
                label = label * (1.0 - self.epsilon) + self.epsilon

        if x.dtype == paddle.float16:
            x = paddle.cast(x, 'float32')
        loss = F.binary_cross_entropy_with_logits(x, label, reduction='none')
        loss = paddle.sum(loss, axis=-1)
        loss = loss.mean()

        return loss
