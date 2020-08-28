#!/usr/bin/python
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from distribute_base import FleetDistRunnerBase
from argument import params_args
import math


class DeepFM(FleetDistRunnerBase):

    def input_data(self, params):
        num_field = params.num_field
        raw_feat_idx = fluid.layers.data(
            name='feat_idx', shape=[num_field], dtype='int64')
        raw_feat_value = fluid.layers.data(
            name='feat_value', shape=[num_field], dtype='float32')
        label = fluid.layers.data(
            name='label', shape=[1], dtype='float32')  # None * 1
        self.inputs = [raw_feat_idx, raw_feat_value, label]
        return self.inputs

    def net(self, inputs, params):
        init_value_ = 0.1
        raw_feat_idx = inputs[0]
        raw_feat_value = inputs[1]
        label = inputs[2]
        feat_idx = fluid.layers.reshape(raw_feat_idx,
                                        [-1, params.num_field, 1])  # None * num_field * 1
        feat_value = fluid.layers.reshape(
            raw_feat_value, [-1, params.num_field, 1])  # None * num_field * 1

        # -------------------- first order term  --------------------

        first_weights = fluid.layers.embedding(
            input=feat_idx,
            dtype='float32',
            size=[params.num_feat + 1, 1],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_),
                regularizer=fluid.regularizer.L1DecayRegularizer(
                    params.reg)))  # None * num_field * 1
        y_first_order = fluid.layers.reduce_sum((first_weights * feat_value), 1)

        # -------------------- second order term  --------------------

        feat_embeddings = fluid.layers.embedding(
            input=feat_idx,
            dtype='float32',
            size=[params.num_feat + 1, params.embedding_size],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_ / math.sqrt(float(
                        params.embedding_size)))))  # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value  # None * num_field * embedding_size

        # sum_square part
        summed_features_emb = fluid.layers.reduce_sum(feat_embeddings,
                                                      1)  # None * embedding_size
        summed_features_emb_square = fluid.layers.square(
            summed_features_emb)  # None * embedding_size

        # square_sum part
        squared_features_emb = fluid.layers.square(
            feat_embeddings)  # None * num_field * embedding_size
        squared_sum_features_emb = fluid.layers.reduce_sum(
            squared_features_emb, 1)  # None * embedding_size

        y_second_order = 0.5 * fluid.layers.reduce_sum(
            summed_features_emb_square - squared_sum_features_emb, 1,
            keep_dim=True)  # None * 1

        # -------------------- DNN --------------------

        y_dnn = fluid.layers.reshape(feat_embeddings,
                                     [-1, params.num_field * params.embedding_size])
        for s in params.layer_sizes:
            y_dnn = fluid.layers.fc(
                input=y_dnn,
                size=s,
                act=params.act,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_ / math.sqrt(float(10)))),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_)))
        y_dnn = fluid.layers.fc(
            input=y_dnn,
            size=1,
            act=None,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)))

        # ------------------- DeepFM ------------------

        predict = fluid.layers.sigmoid(y_first_order + y_second_order + y_dnn)
        cost = fluid.layers.log_loss(input=predict, label=label)
        batch_cost = fluid.layers.reduce_sum(cost)

        # for auc
        predict_2d = fluid.layers.concat([1 - predict, predict], 1)
        label_int = fluid.layers.cast(label, 'int64')
        auc_var, batch_auc_var, auc_states = fluid.layers.auc(input=predict_2d,
                                                              label=label_int,
                                                              slide_steps=0)
        return batch_cost, auc_var

    def py_reader(self, params):
        py_reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                          feed_list=self.inputs,
                                                          name='py_reader',
                                                          use_double_buffer=False)

        return py_reader

    def dataset_reader(self, inputs, params):
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(self.inputs)
        pipe_command = "python dataset_generator.py"
        dataset.set_pipe_command(pipe_command)
        dataset.set_batch_size(params.batch_size)
        thread_num = int(params.cpu_num)
        dataset.set_thread(thread_num)
        return dataset


if __name__ == '__main__':
    params = params_args()
    model = DeepFM()
    model.runtime_main(params)
