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
    #
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

        # label = fluid.layers.data(name="label", shape=[1], dtype="int64")

        # inputs = [dense_input] + sparse_input_ids + [label]
        inputs = [dense_input] + sparse_input_ids
        return inputs

    def net(self, input, args, fc_sizes):
        "Dynamic network -> Static network"
        dnn_model = DNNLayer(args.sparse_feature_dim,
                             args.embedding_size, args.dense_feature_dim,
                             len(input[1:]), fc_sizes)

        pos_score, neg_score = dnn_model.forward(input[1:], input[0])

        with fluid.device_guard("gpu"):
            self.pairwise_hinge_loss = PairwiseHingeLoss()
            cost = self.pairwise_hinge_loss.forward(pos_score,neg_score)
            # predict_2d = paddle.nn.functional.softmax(raw_predict_2d)

            # self.predict = predict_2d

            # auc, batch_auc, _ = paddle.fluid.layers.auc(input=self.predict,
            #                                             label=input[-1],
            #                                             num_thresholds=2**12,
            #                                             slide_steps=20)

            # cost = paddle.nn.functional.cross_entropy(
            #     input=raw_predict_2d, label=input[-1])
            avg_cost = fluid.layers.reduce_mean(cost)
            #avg_cost = paddle.mean(x=cost)
            #self.cost = avg_cost
            #self.infer_target_var = avg_cost
            
            # sync_mode = self.config.get("static_benchmark.sync_mode")
            # if sync_mode == "heter":
            #     fluid.layers.Print(auc, message="AUC")
        return avg_cost, None
        #return {'cost': avg_cost}

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
        super(DNNLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        self.user_embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactorsU",
                initializer=paddle.nn.initializer.Uniform()))
        self.item_embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactorsI",
                initializer=paddle.nn.initializer.Uniform()))

        sizes = [sparse_feature_dim * 12] + self.layer_sizes
        acts = ["relu" for _ in range(len(self.layer_sizes))]
        self._user_layers = []
        for i in range(len(layer_sizes)):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self._user_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self._user_layers.append(act)

        # sizes = [sparse_feature_dim * 13 + dense_feature_dim] + self.layer_sizes
        sizes = [sparse_feature_dim * 14 + dense_feature_dim] + self.layer_sizes
        self._item_layers = []
        for i in range(len(layer_sizes)):
            linear = paddle.nn.Linear(
                    in_features=sizes[i],
                    out_features=sizes[i + 1],
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Normal(
                            std=1.0 / math.sqrt(sizes[i]))))
            self._item_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self._item_layers.append(act)
        
        sizes = [sparse_feature_dim * 20] + self.layer_sizes[1:]
        self._neg_item_layers = []
        for i in range(len(layer_sizes) - 1):
            linear = paddle.nn.Linear(
                    in_features=sizes[i],
                    out_features=sizes[i + 1],
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Normal(
                            std=1.0 / math.sqrt(sizes[i]))))
            self._neg_item_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self._neg_item_layers.append(act)

        self.cos_sim = paddle.nn.CosineSimilarity()

    def forward(self, sparse_inputs, dense_inputs):
        user_embs = []
        item_embs_emb = []
        item_embs = []

        ##########################pos_score###########################################
        with fluid.device_guard("cpu"):
            print("sparse_inputs[:12] len:{}".format(len(sparse_inputs[:12])))
            for s_input in sparse_inputs[:12]:
                emb = self.user_embedding(s_input)
                emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
                user_embs.append(emb)
        with fluid.device_guard("cpu"):
            user_dnn = paddle.concat(x=user_embs, axis=1)

        with fluid.device_guard("cpu"):
            for n_layer in self._user_layers:
                user_dnn = n_layer(user_dnn)
            # fluid.layers.Print(user_dnn, message="user_dnn")
          
            print("sparse_inputs[12:] len:{}".format(len(sparse_inputs[12:])))
        with fluid.device_guard("cpu"):
            for i_input in sparse_inputs[12:]:
                emb_i = self.item_embedding(i_input)
                # item_embs_emb.append(emb_i)
            # print("item_embs_emb[0].shape:",item_embs_emb[0].shape)
            # for emb_i in item_embs_emb:
                emb_i = paddle.reshape(emb_i, shape=[-1, self.sparse_feature_dim])
                item_embs.append(emb_i)
            # item_emb = self.item_embedding(sparse_inputs[13:])
            # print(item_emb.shape)
            # item_dnn = paddle.reshape(item_emb, shape=[-1, item_emb.shape[1] * item_emb.shape[2]])
            # item_dnn = paddle.concat(x=item_embs_emb, axis=1)
            # print("item_dnn.shape:", item_dnn.shape)
            # item_dnn = paddle.reshape(item_dnn, shape=[-1, item_dnn.shape[1]*item_dnn.shape[-1]])
            item_dnn = paddle.concat(x=item_embs + [dense_inputs], axis=1)
            print("item_dnn.shape:",item_dnn.shape)
        with fluid.device_guard("gpu"):
            # item_dnn = paddle.concat(x=item_embs , axis=1)
            for n_layer in self._item_layers:
                item_dnn = n_layer(item_dnn)
            
            pos_score = self.cos_sim(item_dnn, user_dnn)
            
            ###################neg_score############################################
            item_prob = F.softmax(item_dnn)

            item_prob_top, item_prob_index = paddle.topk(item_prob, k=20, largest=False, axis=-1)

            item_prob_index_list = paddle.split(item_prob_index, num_or_sections=20, axis=1)
            neg_item_embs = []

        with fluid.device_guard("gpu"):
            for i_input in item_prob_index_list:
                emb_i = self.item_embedding(i_input)
                emb_i = paddle.reshape(emb_i, shape=[-1, self.sparse_feature_dim])
                neg_item_embs.append(emb_i)
            # neg_item_embs = self.item_embedding(item_prob_index)
            # neg_item_dnn = paddle.reshape(neg_item_embs, shape=[-1, neg_item_embs.shape[1] * neg_item_embs.shape[2]])
            neg_item_dnn = paddle.concat(x=neg_item_embs , axis=1)
            print(neg_item_dnn.shape)
            for n_layer in self._neg_item_layers:
                neg_item_dnn = n_layer(neg_item_dnn)
            neg_score = self.cos_sim(neg_item_dnn, user_dnn)
        return pos_score, neg_score

class PairwiseHingeLoss(object):
    def __init__(self, margin=0.8):
        self.margin = margin

    def forward(self, pos, neg):
        loss_part1 = fluid.layers.elementwise_sub(
            fluid.layers.fill_constant_batch_size_like(
                input=pos, shape=[-1, 1], value=self.margin, dtype='float32'),
            pos)
        loss_part2 = fluid.layers.elementwise_add(loss_part1, neg)
        loss_part3 = fluid.layers.elementwise_max(
            fluid.layers.fill_constant_batch_size_like(
                input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'),
            loss_part2)
        return loss_part3
