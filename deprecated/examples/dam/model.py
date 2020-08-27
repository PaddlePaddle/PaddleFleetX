#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
Deep Attention Matching Network
"""

import six
import numpy as np
import paddle.fluid as fluid
import layers
from paddle.fluid.incubate.fleet.collective import fleet
from dataloader import sample_generator
try:
    import cPickle as pickle  #python 2
except ImportError as e:
    import pickle  #python 3

class DAM(object):
    """
    Deep attention matching network
    """

    def __init__(self, exe, max_turn_num, max_turn_len, vocab_size, emb_size,
                 stack_num, channel1_num, channel2_num):
        """
        Init
        """
        self._max_turn_num = max_turn_num
        self._max_turn_len = max_turn_len
        self._vocab_size = vocab_size
        self._emb_size = emb_size
        self._stack_num = stack_num
        self._channel1_num = channel1_num
        self._channel2_num = channel2_num
        self._feed_names = []
        self.word_emb_name = "shared_word_emb"
        self.use_stack_op = True
        self.use_mask_cache = True
        self.use_sparse_embedding = True
    
        self.feed_vars = self.create_data_layers()
        loss, logits = self.create_network()
        loss.persistable = True
        logits.persistable = True
        self.loss = loss
        self.logits = logits

    def init_emb_from_file(self, word_emb_init, place):
        print("start loading word embedding init ...")
        if six.PY2:
            word_emb = np.array(
                    pickle.load(open(
                        word_emb_init, 'rb'))).astype('float32')
        else:
            word_emb = np.array(
                    pickle.load(open(
                        word_emb_init, 'rb'), 
                        encoding="bytes")).astype('float32')
        self.set_word_embedding(word_emb, place)
        print("finish init word embedding  ...")

    def get_loader_from_filelist(self, batch_size, dict_path,
            data_source, filelists, worker_num, worker_index):
        data_conf = {
            "max_turn_num": self._max_turn_num,
            "max_turn_len": self._max_turn_len,
        }
        
        def _dist_wrapper(generator, is_test):
            def _wrapper():
                rank = worker_index
                nranks = worker_num
                for idx, sample in enumerate(generator()):
                    if idx % nranks == rank:
                        yield sample
        
            def _test_wrapper():
                rank = worker_index
                nranks = worker_num
                for idx, sample in enumerate(generator()):
                    if idx // 10 % nranks == rank:
                        yield sample

            return _wrapper if not is_test else _test_wrapper

        loaders = {}
        for data_type, filelist in filelists.items():
            loader = fluid.io.DataLoader.from_generator(
                    feed_list=self.feed_vars,
                    capacity=8,
                    iterable=True)
            generator = sample_generator(
                    filelist,
                    dict_path,
                    data_conf,
                    data_source)
            if data_type == "train":
                generator = _dist_wrapper(generator, is_test=False)
                loader.set_sample_generator(
                        generator,
                        batch_size=batch_size,
                        drop_last=True,
                        places=fluid.cpu_places())
            elif data_type in ("valid", "test"):
                generator = _dist_wrapper(generator, is_test=True)
                loader.set_sample_generator(
                        generator,
                        batch_size=batch_size,
                        drop_last=False,
                        places=fluid.cpu_places())
            loaders[data_type] = loader
        return loaders

    def create_data_layers(self):
        """
        Create data layer
        """
        self._feed_names = []

        self.turns_data = []
        for i in six.moves.xrange(self._max_turn_num):
            name = "turn_%d" % i
            turn = fluid.layers.data(
                name=name, shape=[self._max_turn_len, 1], dtype="int64")
            self.turns_data.append(turn)
            self._feed_names.append(name)

        self.turns_mask = []
        for i in six.moves.xrange(self._max_turn_num):
            name = "turn_mask_%d" % i
            turn_mask = fluid.layers.data(
                name=name, shape=[self._max_turn_len, 1], dtype="float32")
            self.turns_mask.append(turn_mask)
            self._feed_names.append(name)

        self.response = fluid.layers.data(
            name="response", shape=[self._max_turn_len, 1], dtype="int64")
        self.response_mask = fluid.layers.data(
            name="response_mask",
            shape=[self._max_turn_len, 1],
            dtype="float32")
        self.label = fluid.layers.data(name="label", shape=[1], dtype="float32")
        self._feed_names += ["response", "response_mask", "label"]
        return self.turns_data + self.turns_mask + [self.response] \
                + [self.response_mask] + [self.label]

    def get_feed_names(self):
        """
        Return feed names
        """
        return self._feed_names

    def set_word_embedding(self, word_emb, place):
        """
        Set word embedding
        """
        word_emb_param = fluid.global_scope().find_var(
            self.word_emb_name).get_tensor()
        word_emb_param.set(word_emb, place)

    def create_network(self):
        """
        Create network
        """
        mask_cache = dict() if self.use_mask_cache else None

        response_emb = fluid.layers.embedding(
            input=self.response,
            size=[self._vocab_size + 1, self._emb_size],
            is_sparse=self.use_sparse_embedding,
            param_attr=fluid.ParamAttr(
                name=self.word_emb_name,
                initializer=fluid.initializer.Normal(scale=0.1)))

        # response part
        Hr = response_emb
        Hr_stack = [Hr]

        for index in six.moves.xrange(self._stack_num):
            Hr = layers.block(
                name="response_self_stack" + str(index),
                query=Hr,
                key=Hr,
                value=Hr,
                d_key=self._emb_size,
                q_mask=self.response_mask,
                k_mask=self.response_mask,
                mask_cache=mask_cache)
            Hr_stack.append(Hr)

        # context part
        sim_turns = []
        for t in six.moves.xrange(self._max_turn_num):
            Hu = fluid.layers.embedding(
                input=self.turns_data[t],
                size=[self._vocab_size + 1, self._emb_size],
                is_sparse=self.use_sparse_embedding,
                param_attr=fluid.ParamAttr(
                    name=self.word_emb_name,
                    initializer=fluid.initializer.Normal(scale=0.1)))
            Hu_stack = [Hu]

            for index in six.moves.xrange(self._stack_num):
                # share parameters
                Hu = layers.block(
                    name="turn_self_stack" + str(index),
                    query=Hu,
                    key=Hu,
                    value=Hu,
                    d_key=self._emb_size,
                    q_mask=self.turns_mask[t],
                    k_mask=self.turns_mask[t],
                    mask_cache=mask_cache)
                Hu_stack.append(Hu)

            # cross attention
            r_a_t_stack = []
            t_a_r_stack = []
            for index in six.moves.xrange(self._stack_num + 1):
                t_a_r = layers.block(
                    name="t_attend_r_" + str(index),
                    query=Hu_stack[index],
                    key=Hr_stack[index],
                    value=Hr_stack[index],
                    d_key=self._emb_size,
                    q_mask=self.turns_mask[t],
                    k_mask=self.response_mask,
                    mask_cache=mask_cache)
                r_a_t = layers.block(
                    name="r_attend_t_" + str(index),
                    query=Hr_stack[index],
                    key=Hu_stack[index],
                    value=Hu_stack[index],
                    d_key=self._emb_size,
                    q_mask=self.response_mask,
                    k_mask=self.turns_mask[t],
                    mask_cache=mask_cache)

                t_a_r_stack.append(t_a_r)
                r_a_t_stack.append(r_a_t)

            t_a_r_stack.extend(Hu_stack)
            r_a_t_stack.extend(Hr_stack)

            if self.use_stack_op:
                t_a_r = fluid.layers.stack(t_a_r_stack, axis=1)
                r_a_t = fluid.layers.stack(r_a_t_stack, axis=1)
            else:
                for index in six.moves.xrange(len(t_a_r_stack)):
                    t_a_r_stack[index] = fluid.layers.unsqueeze(
                        input=t_a_r_stack[index], axes=[1])
                    r_a_t_stack[index] = fluid.layers.unsqueeze(
                        input=r_a_t_stack[index], axes=[1])

                t_a_r = fluid.layers.concat(input=t_a_r_stack, axis=1)
                r_a_t = fluid.layers.concat(input=r_a_t_stack, axis=1)

            # sim shape: [batch_size, 2*(stack_num+1), max_turn_len, max_turn_len]
            sim = fluid.layers.matmul(
                x=t_a_r, y=r_a_t, transpose_y=True, alpha=1 / np.sqrt(200.0))
            sim_turns.append(sim)

        if self.use_stack_op:
            sim = fluid.layers.stack(sim_turns, axis=2)
        else:
            for index in six.moves.xrange(len(sim_turns)):
                sim_turns[index] = fluid.layers.unsqueeze(
                    input=sim_turns[index], axes=[2])
            # sim shape: [batch_size, 2*(stack_num+1), max_turn_num, max_turn_len, max_turn_len]
            sim = fluid.layers.concat(input=sim_turns, axis=2)

        final_info = layers.cnn_3d(sim, self._channel1_num, self._channel2_num)
        loss, logits = layers.loss(final_info, self.label)
        return loss, logits
