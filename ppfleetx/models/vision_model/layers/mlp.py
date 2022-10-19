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
from .initializer import xavier_uniform_, mlp_bias_normal_

__all__ = [
    'ViTMLP',
    'MoCoV2Projector',
    'MoCoV3ProjectorOrPredictor',
]


class ViTMLP(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_uniform_(m.weight)
            mlp_bias_normal_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MoCoV2Projector(nn.Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def forward(self, x):
        x = self.mlp(x)
        return x


class MoCoV3ProjectorOrPredictor(nn.Layer):
    def __init__(self, num_layers, in_dim, mlp_dim, out_dim, last_bn=True):
        super().__init__()

        mlp = []
        for l in range(num_layers):
            dim1 = in_dim if l == 0 else mlp_dim
            dim2 = out_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU())
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(
                    nn.BatchNorm1d(
                        dim2, weight_attr=False, bias_attr=False))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        x = self.mlp(x)
        return x
