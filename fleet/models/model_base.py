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

import os
import paddle.fluid as fluid
from input_maker import InputMaker
import math
import logging
logging.basicConfig()

# to define a model
# 1) implement build_train_net
# 2) define the reader for the model
# 3) implement get_loss, get_metrics, get_dataset_input

class MultiSlotDNNCTR(object):
    def __init__(self):
        self.slot_filename = None
        self.hidden_layers = [1024, 512, 512, 512, 128]
        self.emb_dim = 9
        self.dict_size = 100000001
        self.loss = None
        self.metrics = {}
        self.input_slots = []
        self.label_name = "label"
        self.label = None
        self.data_generator_file = None
        self.logger = logging.getLogger("MultiSlotCTR")

    def set_data_generator_file(self, data_generator_file):
        self.data_generator_file = data_generator_file

    def get_input_vars(self):
        return [self.label] + self.input_slots

    def get_pipe_command(self):
        if self.data_generator_file == None:
            self.logger.error("You should call set_data_generator_file"
                              "first to use dataset for training")
            exit(-1)
        if not os.path.exists(self.data_generator_file):
            self.logger.error("Your data_generator_file does not exist")
            exit(-1)
        cmd = "python {} {} {}".format(
            self.data_generator_file, self.slot_filename,
            self.dict_size)
        return cmd

    def get_loss(self):
        return self.loss

    def get_metrics(self):
        return self.metrics

    def build_train_net(self, **kwargs):
        self.slot_filename = kwargs.get("slot_filename", "")
        self.emb_dim = kwargs.get("emb_dim", 9)
        self.dict_size = kwargs.get("dict_size", 100000001)
        
        if self.slot_filename == "":
            self.logger.error("You should assign slot_filename to define input")
            exit(-1)
        if not os.path.exists(self.slot_filename):
            self.logger.error("Your slot file does not exist")
            exit(-1)
        
        input_maker = InputMaker()
        with open(self.slot_filename) as fin:
            slot_names = fin.readlines()
        for slot in slot_names:
            self.input_slots.append(input_maker.sparse(name=slot))

        self.label = input_maker.dense(name=self.label_name, shape=[1])

        def embedding_layer(input):
            emb = fluid.layers.embedding(
                input=input,
                is_sparse=True,
                is_distributed=False,
                size=[self.dict_size, self.emb_dim],
                param_attr=fluid.ParamAttr(
                    name="Emb",
                    initializer=fluid.initializer.Uniform()))
            emb_sum = fluid.layers.sequence_pool(
                input=emb, pool_type='sum')
            return emb_sum
                
        def fc(input, output_size):
            output = fluid.layers.fc(
                input=input, size=output_size,
                act='relu', param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Normal(
                        scale=1.0 / math.sqrt(input.shape[1]))))
            return output

        emb_sums = list(map(embedding_layer, self.input_slots))
        concated = fluid.layers.concat(emb_sums, axis=1)
        fc_list = [concated]

        for size in self.hidden_layers:
            fc_list.append(fc(fc_list[-1], size))

        predict = fluid.layers.fc(input=fc_list[-1], size=2, act='softmax',
                                  param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                      scale=1.0 / math.sqrt(fc_list[-1].shape[1]))))
        cost = fluid.layers.cross_entropy(input=predict, label=self.label)
        avg_cost = fluid.layers.reduce_sum(cost)
        accuracy = fluid.layers.accuracy(input=predict, label=self.label)
        auc_var, batch_auc_var, auc_states = \
            fluid.layers.auc(input=predict, label=self.label,
                             num_thresholds=2 ** 12, slide_steps=20)
        self.startup_program = fluid.default_startup_program()
        self.main_program = fluid.default_main_program()
        self.loss = avg_cost
        self.metrics["auc"] = auc_var
        self.metrics["batch_auc"] = batch_auc_var



