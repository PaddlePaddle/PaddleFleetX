# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
import paddle.nn as nn
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import math


class CTR(object):
    """
    DNN for Click-Through Rate prediction
    """

    #def __init__(self, config):
    #    self.cost = None
    #    self.metrics = {}
    #    self.config = config
    #    self.init_hyper_parameters()

    #def init_hyper_parameters(self):
    #    self.dense_feature_dim = self.config.get(
    #        "hyper_parameters.dense_feature_dim")
    #    self.sparse_feature_dim = self.config.get(
    #        "hyper_parameters.sparse_feature_dim")
    #    self.embedding_size = self.config.get(
    #        "hyper_parameters.embedding_size")
    #    self.fc_sizes = self.config.get(
    #        "hyper_parameters.fc_sizes")

    #    self.learning_rate = self.config.get(
    #        "hyper_parameters.optimizer.learning_rate")
    #    self.adam_lazy_mode = self.config.get(
    #        "hyper_parameters.optimizer.adam_lazy_mode")

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

        label = fluid.layers.data(name="label", shape=[1], dtype="int64")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def net(self, input, args, fc_sizes):
        dnn_model = DNNLayer(args.sparse_feature_dim,
                             args.embedding_size, args.dense_feature_dim,
                             len(input[1:-1]), fc_sizes)

        print("len input: {}   input[1:-1]: {}".format(len(input), len(input[1:-1])))

        raw_predict_2d = dnn_model.forward(input[1:-1], input[0])

        with fluid.device_guard("gpu"):
            predict_2d = fluid.layers.fc(
                input=raw_predict_2d,
                size=2,
                act="softmax",
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                    scale=1 / math.sqrt(raw_predict_2d.shape[1]))),
            )

            label_ = fluid.layers.cast(input[-1], dtype="int64")

            auc, _, _ = paddle.fluid.layers.auc(input=predict_2d,
                                                label=label_,
                                                num_thresholds=2**12,
                                                slide_steps=20)

            cost = fluid.layers.cross_entropy(input=predict_2d, label=label_)
            avg_cost = fluid.layers.reduce_mean(cost)
            
            #fluid.layers.Print(auc, message="training auc_var")
            #fluid.layers.Print(avg_cost, message="training avg_cost")
            #fluid.layers.Print(predict_2d, message="training predict")

        return avg_cost, auc  



    #def minimize(self, strategy=None):
    #    optimizer = fluid.optimizer.SGD(
    #        self.learning_rate)
    #    if strategy != None:
    #        optimizer = fleet.distributed_optimizer(optimizer, strategy)
    #    optimizer.minimize(self.cost)


"""
This file come from PaddleRec/models/rank/dnn/dnn_net.py
"""


class DNNLayer:
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        #super(DNNLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.embedding_g = paddle.nn.Embedding(
            100,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactorsG",
                initializer=paddle.nn.initializer.Uniform()))

        sizes = [sparse_feature_dim * num_field + dense_feature_dim
                 ] + self.layer_sizes
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes)):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))

            #self.add_sublayer('linear_%d' % i, linear)

            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                #self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

        # small towel
        small_layers = [128]
        sizes = [sparse_feature_dim * 5] + [128]
        acts = ["relu" for _ in range(len(small_layers))]
        self._small_mlp_layers = []
        for i in range(len(small_layers)):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self._small_mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self._small_mlp_layers.append(act)
        
        total_layers = [128 + layer_sizes[-1]]
        sizes = [128 + layer_sizes[-1]] + [2]
        self._total_mlp_layers = []
        for i in range(len(total_layers)):
            linear = paddle.nn.Linear(
                    in_features=sizes[i],
                    out_features=sizes[i + 1],
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Normal(
                            std=1.0 / math.sqrt(sizes[i]))))
            self._total_mlp_layers.append(linear)

    def forward(self, sparse_inputs, dense_inputs):

        sparse_embs = []
        with fluid.device_guard("cpu"):
            for s_input in sparse_inputs:
                emb = self.embedding(s_input)
                emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
                sparse_embs.append(emb)
        with fluid.device_guard("gpu"):
            y_dnn = paddle.concat(x=sparse_embs + [dense_inputs], axis=1)

        with fluid.device_guard("gpu"):
            for n_layer in self._mlp_layers:
                y_dnn = n_layer(y_dnn)

        sparse_embs_g = []
        with fluid.device_guard("gpu"):
            for s_input in sparse_inputs[0:5]:
                emb = self.embedding_g(s_input)
                emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
                print("emb shape:{}".format(emb.shape))
                sparse_embs_g.append(emb)
            y_dnn_g = paddle.concat(x=sparse_embs_g, axis=1)
            print("y_dnn_g :{}".format(y_dnn_g.shape))
            for n_layer in self._small_mlp_layers:
                y_dnn_g = n_layer(y_dnn_g)
            
        with fluid.device_guard("gpu"):
            y_dnn_total = paddle.concat(x=[y_dnn] + [y_dnn_g], axis=1)
            for n_layer in self._total_mlp_layers:
                y_dnn_total = n_layer(y_dnn_total)
        return y_dnn_total
