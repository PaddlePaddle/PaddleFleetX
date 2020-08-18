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

import math
import paddle.fluid as fluid
from distribute_base import FleetRunnerBase
from argument import params_args


class Simnet_bow(FleetRunnerBase):
    def input_data(self, params):
        with fluid.unique_name.guard():
            q = fluid.layers.data(
                name="query", shape=[1], dtype="int64", lod_level=1)
            pt = fluid.layers.data(
                name="pos_title", shape=[1], dtype="int64", lod_level=1)
            nt = fluid.layers.data(
                name="neg_title", shape=[1], dtype="int64", lod_level=1)

            self.inputs = [q, pt, nt]
        return self.inputs

    def net(self, inputs, params):
        with fluid.unique_name.guard():
            is_distributed = False
            is_sparse = True
            query = inputs[0]
            pos = inputs[1]
            neg = inputs[2]
            dict_dim = params.dict_dim
            emb_dim = params.emb_dim
            hid_dim = params.hid_dim
            emb_lr = params.learning_rate * 3
            base_lr = params.learning_rate

            q_emb = fluid.layers.embedding(input=query,
                                           is_distributed=is_distributed,
                                           size=[dict_dim, emb_dim],
                                           param_attr=fluid.ParamAttr(name="__emb__",
                                                                      learning_rate=emb_lr,
                                                                      initializer=fluid.initializer.Xavier()),
                                           is_sparse=is_sparse
                                           )
            # vsum
            q_sum = fluid.layers.sequence_pool(
                input=q_emb,
                pool_type='sum')
            q_ss = fluid.layers.softsign(q_sum)
            # fc layer after conv
            q_fc = fluid.layers.fc(input=q_ss,
                                   size=hid_dim,
                                   param_attr=fluid.ParamAttr(name="__q_fc__", learning_rate=base_lr,
                                                              initializer=fluid.initializer.Xavier()))
            # embedding
            pt_emb = fluid.layers.embedding(input=pos,
                                            is_distributed=is_distributed,
                                            size=[dict_dim, emb_dim],
                                            param_attr=fluid.ParamAttr(name="__emb__", learning_rate=emb_lr,
                                                                       initializer=fluid.initializer.Xavier()),
                                            is_sparse=is_sparse)
            # vsum
            pt_sum = fluid.layers.sequence_pool(
                input=pt_emb,
                pool_type='sum')
            pt_ss = fluid.layers.softsign(pt_sum)
            # fc layer
            pt_fc = fluid.layers.fc(input=pt_ss,
                                    size=hid_dim,
                                    param_attr=fluid.ParamAttr(name="__fc__", learning_rate=base_lr,
                                                               initializer=fluid.initializer.Xavier()),
                                    bias_attr=fluid.ParamAttr(name="__fc_b__",
                                                              initializer=fluid.initializer.Xavier()))

            # embedding
            nt_emb = fluid.layers.embedding(input=neg,
                                            is_distributed=is_distributed,
                                            size=[dict_dim, emb_dim],
                                            param_attr=fluid.ParamAttr(name="__emb__",
                                                                       learning_rate=emb_lr,
                                                                       initializer=fluid.initializer.Xavier()),
                                            is_sparse=is_sparse)

            # vsum
            nt_sum = fluid.layers.sequence_pool(
                input=nt_emb,
                pool_type='sum')
            nt_ss = fluid.layers.softsign(nt_sum)
            # fc layer
            nt_fc = fluid.layers.fc(input=nt_ss,
                                    size=hid_dim,
                                    param_attr=fluid.ParamAttr(name="__fc__", learning_rate=base_lr,
                                                               initializer=fluid.initializer.Xavier()),
                                    bias_attr=fluid.ParamAttr(name="__fc_b__",
                                                              initializer=fluid.initializer.Xavier()))
            cos_q_pt = fluid.layers.cos_sim(q_fc, pt_fc)
            cos_q_nt = fluid.layers.cos_sim(q_fc, nt_fc)
            # loss
            avg_cost = self.get_loss(cos_q_pt, cos_q_nt,params)
            # acc
            acc = self.get_acc(cos_q_nt, cos_q_pt,params)

            return avg_cost, acc, cos_q_pt

    def get_acc(self,cos_q_nt, cos_q_pt,params):
        cond = fluid.layers.less_than(cos_q_nt, cos_q_pt)
        cond = fluid.layers.cast(cond, dtype='float64')
        cond_3 = fluid.layers.reduce_sum(cond)
        acc = fluid.layers.elementwise_div(cond_3,
                                           fluid.layers.fill_constant(shape=[1], value=params.batch_size * 1.0,
                                                                      dtype='float64'),
                                           name="simnet_acc")
        return acc

    def get_loss(self,cos_q_pt, cos_q_nt,params):
        loss_op1 = fluid.layers.elementwise_sub(
            fluid.layers.fill_constant_batch_size_like(input=cos_q_pt, shape=[-1, 1], value=params.margin,
                                                       dtype='float32'), cos_q_pt)
        loss_op2 = fluid.layers.elementwise_add(loss_op1, cos_q_nt)
        loss_op3 = fluid.layers.elementwise_max(
            fluid.layers.fill_constant_batch_size_like(input=loss_op2, shape=[-1, 1], value=0.0,
                                                       dtype='float32'), loss_op2)
        avg_cost = fluid.layers.mean(loss_op3)
        return avg_cost

    def py_reader(self, params):
        data_shapes = []
        data_lod_levels = []
        data_types = []

        # query ids
        data_shapes.append((-1, 1))
        data_lod_levels.append(1)
        data_types.append('int64')
        # pos_title_ids
        data_shapes.append((-1, 1))
        data_lod_levels.append(1)
        data_types.append('int64')
        # neg_title_ids
        data_shapes.append((-1, 1))
        data_lod_levels.append(1)
        data_types.append('int64')
        # label
        data_shapes.append((-1, 1))
        data_lod_levels.append(1)
        data_types.append('int64')

        reader = fluid.layers.py_reader(capacity=64,
                                        shapes=data_shapes,
                                        lod_levels=data_lod_levels,
                                        dtypes=data_types,
                                        name='py_reader',
                                        use_double_buffer=False)

        return reader

    def dataset_reader(self, inputs, params):
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_batch_size(params.batch_size)
        dataset.set_use_var([inputs[0], inputs[1], inputs[2]])
        dataset.set_batch_size(params.batch_size)
        pipe_command = 'python dataset_generator.py'
        dataset.set_pipe_command(pipe_command)
        dataset.set_thread(int(params.cpu_num))
        return dataset


if __name__ == '__main__':
    params = params_args()
    model = Simnet_bow()
    model.runtime_main(params)
