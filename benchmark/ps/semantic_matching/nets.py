#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

const_var=fluid.layers.fill_constant_batch_size_like
param_config=fluid.ParamAttr
pooling=fluid.layers.sequence_pool
elem_sub=fluid.layers.elementwise_sub
elem_add=fluid.layers.elementwise_add

def get_global_pn(pos_score, neg_score):
    wrong = fluid.layers.cast(
        fluid.layers.less_than(pos_score, neg_score), dtype='float32')
    wrong_cnt = fluid.layers.reduce_sum(wrong)
    right = fluid.layers.cast(
        fluid.layers.less_than(neg_score, pos_score), dtype='float32')
    right_cnt = fluid.layers.reduce_sum(right)

    global_right_cnt = fluid.default_startup_program().global_block().create_var(
        name="right_cnt", dtype=fluid.core.VarDesc.VarType.FP32, shape=[1], persistable=True,
        initializer=fluid.initializer.Constant(value=float(0), force_cpu=True))

    global_wrong_cnt = fluid.default_startup_program().global_block().create_var(
        name="wrong_cnt", dtype=fluid.core.VarDesc.VarType.FP32, shape=[1], persistable=True,
        initializer=fluid.initializer.Constant(value=float(0), force_cpu=True))
    
    fluid.default_main_program().global_block().create_var(
        name="right_cnt", dtype=fluid.core.VarDesc.VarType.FP32, shape=[1], persistable=True)

    fluid.default_main_program().global_block().create_var(
        name="wrong_cnt", dtype=fluid.core.VarDesc.VarType.FP32, shape=[1], persistable=True)

    #fluid.layers.Print(global_right_cnt)
    #fluid.layers.Print(global_wrong_cnt)
    global_right_cnt.stop_gradient = True
    global_wrong_cnt.stop_gradient = True

    fluid.default_main_program().global_block().append_op(
        type='elementwise_add', inputs={'X':[global_right_cnt], 'Y':[right_cnt]},
        outputs={'Out':[global_right_cnt]})

    fluid.default_main_program().global_block().append_op(
        type='elementwise_add', inputs={'X':[global_wrong_cnt], 'Y':[wrong_cnt]},
        outputs={'Out':[global_wrong_cnt]})

    pn = fluid.layers.elementwise_div(global_right_cnt, global_wrong_cnt)
    return global_right_cnt, global_wrong_cnt, pn

def get_pn(pos_score, neg_score):
    """acc"""
    wrong = fluid.layers.cast(
        fluid.layers.less_than(pos_score, neg_score), dtype='float32')
    wrong_cnt = fluid.layers.reduce_sum(wrong)
    right = fluid.layers.cast(
        fluid.layers.less_than(neg_score, pos_score), dtype='float32')
    right_cnt = fluid.layers.reduce_sum(right)
    pn = fluid.layers.elementwise_div(right_cnt, wrong_cnt)
    return right_cnt, wrong_cnt, pn

def bow_encoder(query, pos_title, neg_title,
                dict_dim, emb_dim, hid_dim,
                emb_lr, fc_lr, margin):
    q_emb = fluid.layers.embedding(
        input=query,
        size=[dict_dim, emb_dim],
        param_attr=param_config(
            name="__emb__",
            learning_rate=emb_lr),
        is_sparse=True)

    pt_emb = fluid.layers.embedding(
        input=pos_title,
        size=[dict_dim, emb_dim],
        param_attr=param_config(
            name="__emb__",
            learning_rate=emb_lr),
        is_sparse=True)
    
    nt_emb = fluid.layers.embedding(
        input=neg_title,
        size=[dict_dim, emb_dim],
        param_attr=param_config(
            name="__emb__",
            learning_rate=emb_lr),
        is_sparse=True)

    q_sum = pooling(input=q_emb, pool_type='sum')
    pt_sum = pooling(input=pt_emb, pool_type='sum')
    nt_sum = pooling(input=nt_emb, pool_type='sum')

    q_ss = fluid.layers.softsign(q_sum)
    pt_ss = fluid.layers.softsign(pt_sum)
    nt_ss = fluid.layers.softsign(nt_sum)

    q_fc = fluid.layers.fc(
        input=q_ss,
        size=hid_dim,
        param_attr=param_config(
            name="__q_fc__",
            learning_rate=fc_lr),
        bias_attr=param_config(
            name="__q_fc_b__"))

    pt_fc = fluid.layers.fc(
        input=pt_ss,
        size=hid_dim,
        param_attr=param_config(
            name="__fc__",
            learning_rate=fc_lr),
        bias_attr=param_config(
            name="__fc_b__"))

    nt_fc = fluid.layers.fc(
        input=nt_ss,
        size=hid_dim,
        param_attr=param_config(
            name="__fc__",
            learning_rate=fc_lr),
        bias_attr=param_config(
            name="__fc_b__"))
    
    cos_q_pt = fluid.layers.cos_sim(q_fc, pt_fc)
    cos_q_nt = fluid.layers.cos_sim(q_fc, nt_fc)

    margin_var = const_var(input=cos_q_pt,
                           shape=[-1, 1],
                           value=margin,
                           dtype='float32')
    
    margin_minus_qt = elem_sub(margin_var, cos_q_pt)
    margin_minus_nt = elem_add(margin_minus_qt, cos_q_nt)
    
    zero_var = const_var(input=margin_minus_nt,
                         shape=[-1, 1],
                         value=0.0,
                         dtype='float32')
    
    loss = fluid.layers.elementwise_max(
        zero_var, margin_minus_nt)

    avg_cost = fluid.layers.mean(loss)
    pnum, nnum, pn = get_global_pn(cos_q_pt, cos_q_nt)

    return avg_cost, cos_q_pt, cos_q_nt, pnum, nnum, pn
