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
"""Ernie model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import six
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.distributed.fleet as fleet
import numpy as np
from functools import partial

from model.transformer_encoder import encoder, pre_process_layer
from model.transformer_encoder import gelu


class ParallelEmbedding(nn.Layer):
    """
    Parallel Embedding
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 topo,
                 weight_attr=None,
                 name=None):
        super(ParallelEmbedding, self).__init__()
        self.rank = topo.mp.rank
        self.world_size = topo.mp.size
        self.num_embeddings = num_embeddings
        self.is_mp = (self.world_size > 1)

        assert num_embeddings % self.world_size == 0, \
            "The length of the vocabulary must be divisible by the parallelism degree of MP"

        per_part_size = num_embeddings // self.world_size

        self.vocab_start_index = self.rank * per_part_size
        self._dtype = self._helper.get_default_dtype()
        self._size = [per_part_size, embedding_dim]
        self._weight_attr = weight_attr
        self._name = name

        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=self._size,
            dtype=self._dtype,
            is_bias=False)
        self.weight.is_distributed = True

        startup_block = paddle.static.default_startup_program().global_block()
        main_block = paddle.static.default_main_program().global_block()
        startup_block.vars[self.weight.name].is_distributed = True
        main_block.vars[self.weight.name].is_distributed = True
        if self.is_mp:
            self.model_group = \
                fleet.get_hybrid_communicate_group().get_model_parallel_group()

    def forward(self, x):
        if self.is_mp:
            output_parallel = paddle.distributed.collective._c_lookup_table(
                self.weight,
                x,
                start_index=self.vocab_start_index,
                name=self._name)
            output = paddle.distributed.collective._mp_allreduce(
                output_parallel,
                group=self.model_group,  # None is ok in static
                use_calc_stream=True,
                use_model_parallel=True)
        else:
            output = paddle.nn.functional.embedding(
                x,
                weight=self.weight,
                padding_idx=None,
                sparse=False,
                name=self._name)
        return output


def parallel_matmul(lm_output, logit_weights, parallel_output, topo):
    world_size = topo.mp.size
    rank = topo.mp.rank

    if world_size > 1:
        model_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_group)

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


class ErnieConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))


class ErnieModel(object):
    def __init__(self,
                 src_ids,
                 sentence_ids,
                 pos_ids,
                 config,
                 task_ids=None,
                 weight_sharing=True,
                 topo=None,
                 device="gpu"):

        self._emb_size = config['emb_size'] if config['emb_mapping_in'] else config['hidden_size']
        self._hidden_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._sent_types = config['sent_type_vocab_size']
        self._task_types = config['task_type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._param_share = config['param_share']
        self._weight_sharing = weight_sharing
        self.config = config
        self.topo = topo
        self.preln = config['preln'] if 'preln' in config.keys() else False
        self.pre_encoder_cmd = "" if self.preln else self.config['pre_encoder_cmd']

        self._checkpoints = []
        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._task_emb_name = "task_embedding"
        self._dtype = "float32"
        self._emb_dtype = "float32"
        self._device = device

        if device == "gpu":
            self._int_type = "int64"
        elif device == "npu":
            self._int_type = "int32"
        else:
            raise ValueError(f"error device: {device}")

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self.src_ids = src_ids
        self.position_ids = pos_ids
        self.sentence_ids = sentence_ids
        self.task_ids = task_ids
        self.input_mask = self._build_input_mask(src_ids)

        self._build_model()

    def _build_model(self, emb=None):
        # padding id in vocabulary must be set to 0
        if emb is None:
            if self.topo is None or self.topo.mp.size == 1:
                src_emb = paddle.nn.Embedding(
                        num_embeddings=self._voc_size,
                        embedding_dim=self._emb_size,
                        sparse=False,
                        weight_attr=fluid.ParamAttr(
                            name=self._word_emb_name, initializer=self._param_initializer))
                emb_out = src_emb(self.src_ids)
            else:
                self._word_emb_name = self._word_emb_name + '_' + str(self.topo.mp.rank)
                src_ids = fluid.layers.squeeze(self.src_ids, [-1])
                if paddle.is_compiled_with_npu():
                    emb_out = paddle.distributed.split(
                        src_ids,
                        size=(self._voc_size, self._emb_size),
                        operation='embedding',
                        weight_attr=fluid.ParamAttr(
                            name=self._word_emb_name, initializer=self._param_initializer),
                        num_partitions=self.topo.mp.size)
                else:
                    self.word_embeddings = ParallelEmbedding(
                        self._voc_size, self._emb_size, self.topo,
                        weight_attr=paddle.ParamAttr(
                            name=self._word_emb_name, initializer=self._param_initializer)
                    )
                    emb_out = self.word_embeddings(src_ids)
        else:
            emb.stop_gradient = True
            emb_out = fluid.layers.gather_nd(emb, self.src_ids)
            emb_out.stop_gradient = False

        position_emb = paddle.nn.Embedding(
            num_embeddings=self._max_position_seq_len,
            embedding_dim=self._emb_size,
            weight_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer))
        self.position_emb_out = position_emb(self.position_ids)

        sent_emb = paddle.nn.Embedding(
            num_embeddings=self._sent_types,
            embedding_dim=self._emb_size,
            weight_attr=fluid.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer))
        self.sent_emb_out = sent_emb(self.sentence_ids)

        """
        self.task_emb_out = fluid.layers.embedding(
            self.task_ids,
            size=[self._task_types, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._task_emb_name, initializer=self._param_initializer))
        """
        sum_emb = emb_out + self.position_emb_out
        sum_emb = sum_emb + self.sent_emb_out
        # print('[ERROR] for debuging not add task_emb out')
        # emb_out = emb_out + task_emb_out

        # for albert shold be n
        # for bert should be nd
        sum_emb = pre_process_layer(
            sum_emb,
            self.config['pre_encoder_cmd'],
            self._prepostprocess_dropout,
            name='pre_encoder',
            epsilon=self.config['epsilon'])

        if self.config['emb_mapping_in']:
            sum_emb = fluid.layers.fc(input=sum_emb,
                          num_flatten_dims=2,
                          size=self._hidden_size,
                          param_attr=fluid.ParamAttr(
                              name='emb_hidden_mapping',
                              initializer=self._param_initializer),
                          bias_attr='emb_hidden_mapping_bias')

        with fluid.device_guard(f'{self._device}:all'):
            self_attn_mask = paddle.matmul(
                x=self.input_mask, y=self.input_mask, transpose_y=True)

            self_attn_mask = fluid.layers.scale(
                x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
            stack_time = self._n_head
            if self.topo.mp.size > 1:
                stack_time = stack_time // self.topo.mp.size
            n_head_self_attn_mask = fluid.layers.stack(
                x=[self_attn_mask] * stack_time, axis=1)
            n_head_self_attn_mask.stop_gradient = True

        self._enc_out, self._checkpoints = encoder(
            enc_input=sum_emb,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._hidden_size // self._n_head,
            d_value=self._hidden_size // self._n_head,
            d_model=self._hidden_size,
            d_inner_hid=self._hidden_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd=self.config['preprocess_cmd'],
            postprocess_cmd=self.config['postprocess_cmd'],
            param_initializer=self._param_initializer,
            name='encoder',
            param_share=self._param_share,
            epsilon=self.config['epsilon'],
            n_layer_per_block=self.config['n_layer_per_block'],
            topo=self.topo,
            preln=self.preln,
            device=self._device)

    def _build_position_ids(self, src_ids):
        d_shape = fluid.layers.shape(src_ids)
        d_seqlen = d_shape[1]
        d_batch = d_shape[0]
        position_ids = fluid.layers.reshape(
            fluid.layers.range(
                0, d_seqlen, 1, dtype='int32'), [1, d_seqlen, 1],
            inplace=True)
        position_ids = fluid.layers.expand(position_ids, [d_batch, 1, 1])
        position_ids = fluid.layers.cast(position_ids, 'int64')
        position_ids.stop_gradient = True
        return position_ids

    def _build_input_mask(self, src_ids):
        with fluid.device_guard(f'{self._device}:all'):
            zero = fluid.layers.fill_constant([1], dtype=self._int_type, value=0)
            input_mask = fluid.layers.logical_not(fluid.layers.equal(src_ids, zero))  # assume pad id == 0
            input_mask = fluid.layers.cast(input_mask, 'float32')
            input_mask = fluid.layers.unsqueeze(input_mask, [-1])
            input_mask.stop_gradient = True
            return input_mask

    def get_sequence_output(self):
        return self._enc_out

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = fluid.layers.slice(
            input=self._enc_out, axes=[1], starts=[0], ends=[1])

        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._hidden_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc.b_0")
        return next_sent_feat


    def get_next_sentence_output(self, labels):
        next_sent_feat = self.get_pooled_output()
        next_sent_fc_out = fluid.layers.fc(
            input=next_sent_feat,
            num_flatten_dims=1,
            size=33,
            param_attr=fluid.ParamAttr(
                name="next_sent_fc.w_0", initializer=self._param_initializer),
            bias_attr="next_sent_fc.b_0")
        if self._device == "gpu":
            next_sent_fc_out = fluid.layers.reshape(
                next_sent_fc_out, [-1, 33], inplace=True)
        elif self._device == "npu":
            next_sent_fc_out = fluid.layers.reshape(
                next_sent_fc_out, [-1, 33])
        next_sent_loss, next_sent_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=next_sent_fc_out, label=labels, return_softmax=True)
        next_sent_acc = fluid.layers.accuracy(
            input=next_sent_softmax, label=labels)
        mean_next_sent_loss = fluid.layers.mean(next_sent_loss, "mean_next_sent_loss")
        return next_sent_acc, mean_next_sent_loss

    #def get_word_order_output(self, word_label, pos):
    #    pos = fluid.layers.cast(x=pos, dtype='int32')

    #    reshaped_emb_out = fluid.layers.reshape(
    #        x=self._enc_out, shape=[-1, self._hidden_size])

    #    # extract masked tokens' feature
    #    mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=pos)
    #    if self._dtype == "float16":
    #        mask_feat = fluid.layers.cast(x=mask_feat, dtype=self._emb_dtype)

    #    # transform: fc
    #    if self._hidden_act == 'gelu' or self._hidden_act == 'gelu.precise':
    #        _hidden_act = 'gelu'
    #    else:
    #        _hidden_act = None

    #    mask_trans_feat = fluid.layers.fc(
    #        input=mask_feat,
    #        size=self._emb_size,
    #        act=_hidden_act,
    #        param_attr=fluid.ParamAttr(
    #            name='word_order_trans_fc.w_0',
    #            initializer=self._param_initializer),
    #        bias_attr=fluid.ParamAttr(name='word_order_lm_trans_fc.b_0'))

    #    if self._hidden_act == 'gelu' or self._hidden_act == 'gelu.precise':
    #        pass
    #    else:
    #        mask_trans_feat = gelu(mask_trans_feat)

    #    # transform: layer norm 
    #    mask_trans_feat = fluid.layers.layer_norm(
    #        mask_trans_feat,
    #        begin_norm_axis=len(mask_trans_feat.shape) - 1,
    #        param_attr=fluid.ParamAttr(
    #            name='word_order_lm_trans_layer_norm_scale',
    #            initializer=fluid.initializer.Constant(1.)),
    #        bias_attr=fluid.ParamAttr(
    #            name='word_order_lm_trans_layer_norm_bias',
    #            initializer=fluid.initializer.Constant(1.)),
    #        epsilon=self.config['epsilon'])

    #    mask_lm_out_bias_attr = fluid.ParamAttr(
    #        name="word_order_lm_out_fc.b_0",
    #        initializer=fluid.initializer.Constant(value=0.0))

    #    if self._weight_sharing:
    #        fc_out = fluid.layers.matmul(
    #            x=mask_trans_feat,
    #            y=fluid.default_main_program().global_block().var(
    #                self._word_emb_name),
    #            transpose_y=True)
    #        fc_out += fluid.layers.create_parameter(
    #            shape=[self._voc_size],
    #            dtype=self._emb_dtype,
    #            attr=mask_lm_out_bias_attr,
    #            is_bias=True)

    #    else:
    #        fc_out = fluid.layers.fc(input=mask_trans_feat,
    #                                 size=self._voc_size,
    #                                 param_attr=fluid.ParamAttr(
    #                                     name="word_order_lm_out_fc.w_0",
    #                                     initializer=self._param_initializer),
    #                                 bias_attr=mask_lm_out_bias_attr)

    #    ##############
    #    # WORD_ORDER
    #    word_order_ce_loss = fluid.layers.softmax_with_cross_entropy(logits=fc_out, label=word_label)
    #    mean_word_order_loss = fluid.layers.mean(word_order_ce_loss)

    #    return mean_word_order_loss

    def get_lm_output(self, mask_label, mask_pos):
        """Get the loss & accuracy for pretraining"""
        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        # extract the first token feature in each sentence
        reshaped_emb_out = fluid.layers.reshape(
            x=self._enc_out, shape=[-1, self._hidden_size])

        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)
        if self._dtype == "float16":
            mask_feat = fluid.layers.cast(x=mask_feat, dtype=self._emb_dtype)

        # transform: fc
        if self._hidden_act == 'gelu' or self._hidden_act == 'gelu.precise':
            _hidden_act = 'gelu'
        else:
            _hidden_act = None

        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=self._emb_size,
            act=_hidden_act,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_fc.w_0',
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))

        if self._hidden_act == 'gelu' or self._hidden_act == 'gelu.precise':
            pass
        else:
            mask_trans_feat = gelu(mask_trans_feat)


        # transform: layer norm 
        mask_trans_feat = fluid.layers.layer_norm(
            mask_trans_feat,
            begin_norm_axis=len(mask_trans_feat.shape) - 1,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_bias',
                initializer=fluid.initializer.Constant(0.)),
            epsilon=self.config['epsilon'])

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))

        if self._weight_sharing:
            fc_out = paddle.matmul(
                x=mask_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self._word_emb_name),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self._voc_size],
                dtype=self._emb_dtype,
                attr=mask_lm_out_bias_attr,
                is_bias=True)

        else:
            fc_out = fluid.layers.fc(input=mask_trans_feat,
                                     size=self._voc_size,
                                     param_attr=fluid.ParamAttr(
                                         name="mask_lm_out_fc.w_0",
                                         initializer=self._param_initializer),
                                     bias_attr=mask_lm_out_bias_attr)
            # if self.topo is None or self.topo.mp.size == 1:
            #     weight = paddle.create_parameter(shape=[self._voc_size, self._emb_size], dtype=self._emb_dtype)
            # else:
            #     weight = self.word_embeddings.weight
            # weight = paddle.create_parameter(shape=weight.shape, dtype=weight.dtype)
            # # fc_out = parallel_matmul(mask_trans_feat, weight, True, self.topo)
            # fc_out = parallel_matmul(mask_trans_feat, weight, False, self.topo)

        # if self.topo is None or self.topo.mp.size == 1:
        if True:
            loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
            if paddle.is_compiled_with_npu():
                loss_func = fluid.layers.softmax_with_cross_entropy
        else:
            loss_func = fleet.meta_parallel.ParallelCrossEntropy()

        mask_lm_loss = loss_func(fc_out, mask_label)
        mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss, name="mean_mask_lm_loss")

        return mask_lm_loss, mean_mask_lm_loss

    def get_task_output(self, task, task_labels):
        task_fc_out = fluid.layers.fc(
            input=self.next_sent_feat,
            size=task["num_labels"],
            param_attr=fluid.ParamAttr(
                name=task["task_name"] + "_fc.w_0", initializer=self._param_initializer),
            bias_attr=task["task_name"] + "_fc.b_0")
        task_loss, task_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=task_fc_out, label=task_labels, return_softmax=True)
        task_acc = fluid.layers.accuracy(
            input=task_softmax, label=task_labels)
        mean_task_loss = fluid.layers.mean(task_loss)
        return mean_task_loss, task_acc

