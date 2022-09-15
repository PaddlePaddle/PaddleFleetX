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


class TopkAcc(nn.Layer):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]

        if len(label.shape) == 1:
            label = label.reshape([label.shape[0], -1])

        if label.dtype == paddle.int32:
            label = paddle.cast(label, 'int64')
        metric_dict = dict()
        for i, k in enumerate(self.topk):
            acc = paddle.metric.accuracy(x, label, k=k).item()
            metric_dict["top{}".format(k)] = acc
            if i == 0:
                metric_dict["metric"] = acc

        return metric_dict
