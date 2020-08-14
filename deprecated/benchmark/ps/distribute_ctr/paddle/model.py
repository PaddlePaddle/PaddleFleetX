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


class CTR(FleetDistRunnerBase):

    def input_data(self, params):
        with fluid.unique_name.guard():
            dense_feature_dim = params.dense_feature_dim
            self.dense_input = fluid.layers.data(
                name="dense_input", shape=[dense_feature_dim], dtype='float32')
    
            self.sparse_input_ids = [
                fluid.layers.data(name="C" + str(i), shape=[1], lod_level=1, dtype='int64')
                for i in range(1, 27)]
    
            self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    
            self._words = [self.dense_input] + self.sparse_input_ids + [self.label]
        return self._words

    def net(self, inputs, params):
        with fluid.unique_name.guard():
            sparse_feature_dim = params.sparse_feature_dim
            embedding_size = params.embedding_size
    
            words = inputs
    
            def embedding_layer(input):
                return fluid.layers.embedding(
                    input=input,
                    is_sparse=True,
                    is_distributed=False,
                    size=[sparse_feature_dim, embedding_size],
                    param_attr=fluid.ParamAttr(name="SparseFeatFactors",
                                               initializer=fluid.initializer.Uniform()))
    
            sparse_embed_seq = list(map(embedding_layer, words[1:-1]))
            concated = fluid.layers.concat(sparse_embed_seq + words[0:1], axis=1)
    
            fc1 = fluid.layers.fc(input=concated, size=400, act='relu',
                                  param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                      scale=1 / math.sqrt(concated.shape[1]))))
            fc2 = fluid.layers.fc(input=fc1, size=400, act='relu',
                                  param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                      scale=1 / math.sqrt(fc1.shape[1]))))
            fc3 = fluid.layers.fc(input=fc2, size=400, act='relu',
                                  param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                      scale=1 / math.sqrt(fc2.shape[1]))))
            predict = fluid.layers.fc(input=fc3, size=2, act='softmax',
                                      param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                          scale=1 / math.sqrt(fc3.shape[1]))))
    
            cost = fluid.layers.cross_entropy(input=predict, label=words[-1])
            avg_cost = fluid.layers.reduce_sum(cost)
            accuracy = fluid.layers.accuracy(input=predict, label=words[-1])
            auc_var, batch_auc_var, auc_states = \
                fluid.layers.auc(input=predict, label=words[-1], num_thresholds=2 ** 12, slide_steps=20)

        return avg_cost, auc_var, batch_auc_var, words

    def py_reader(self, params):
        py_reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                          feed_list=self._words,
                                                          name='py_reader',
                                                          use_double_buffer=False)

        return py_reader

    def dataset_reader(self, inputs, params):
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var([self.dense_input] + self.sparse_input_ids + [self.label])
        pipe_command = "python dataset_generator.py"
        dataset.set_pipe_command(pipe_command)
        dataset.set_batch_size(params.batch_size)
        thread_num = int(params.cpu_num)
        dataset.set_thread(thread_num)
        return dataset


if __name__ == '__main__':
    params = params_args()
    model = CTR()
    model.runtime_main(params)
