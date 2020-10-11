#!/usr/bin/python
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import math


class CTR(object):
    """
    DNN for Click-Through Rate prediction
    """

    def input_data(self, args):
        dense_input = fluid.layers.data(name="dense_input",
                                        shape=[args.dense_feature_dim],
                                        dtype="float32")

        sparse_input_ids = [
            fluid.layers.data(name="C" + str(i),
                              shape=[1],
                              lod_level=1,
                              dtype="int64") for i in range(1, 27)
        ]

        label = fluid.layers.data(name="label", shape=[1], dtype="float32")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def net(self, inputs, args):

        sparse_inputs = []
        for sparse_input in inputs[1:-1]:
            sparse_input = fluid.layers.reshape(sparse_input, [-1, 1])
            sparse_inputs.append(sparse_input)

        def embedding_layer(input):
            return fluid.layers.embedding(
                input=input,
                is_sparse=True,
                size=[args.sparse_feature_dim, args.embedding_size],
                param_attr=fluid.ParamAttr(
                    name="SparseFeatFactors",
                    initializer=fluid.initializer.Uniform()),
            )
        sparse_embed_seq = list(map(embedding_layer, sparse_inputs))

        concated = fluid.layers.concat(
            sparse_embed_seq + inputs[0:1], axis=1)

        with fluid.device_guard("gpu"):
            fc1 = fluid.layers.fc(
                input=concated,
                size=400,
                act="relu",
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                    scale=1 / math.sqrt(concated.shape[1]))), name="fc1"
            )
            fc2 = fluid.layers.fc(
                input=fc1,
                size=400,
                act="relu",
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                    scale=1 / math.sqrt(fc1.shape[1]))), name="fc2"
            )
            fc3 = fluid.layers.fc(
                input=fc2,
                size=400,
                act="relu",
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                    scale=1 / math.sqrt(fc2.shape[1]))), name="fc3"
            )
            predict = fluid.layers.fc(
                input=fc3,
                size=2,
                act="softmax",
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                    scale=1 / math.sqrt(fc3.shape[1]))),
            )
            label = fluid.layers.cast(inputs[-1], dtype="int64")
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.reduce_sum(cost)
            auc_var, _, _ = fluid.layers.auc(input=predict,
                                             label=label,
                                             num_thresholds=2**12,
                                             slide_steps=20)
            fluid.layers.Print(auc_var, message="training auc_var")

        return avg_cost, auc_var
