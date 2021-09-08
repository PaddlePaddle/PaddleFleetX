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

from __future__ import print_function

import collections
import math
import unittest
import random
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
import paddle.utils as utils
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.framework import in_dygraph_mode
from paddle.nn.layer.transformer import _convert_param_attr_to_list
from paddle.fluid.initializer import Normal, Constant, NumpyArrayInitializer
from paddle.distributed.fleet import fleet
import paddle.static as static
from paddle.distributed import fleet
from args import parse_args
import paddlenlp
import global_setting

paddle.enable_static()


def gpt_pretrain_forward(args, train_program, start_program, topo):
    from modeling import GPTModel, GPTForPretraining, GPTPretrainingCriterion
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        batch_size =  args.global_batch_size // args.dp_degree
        sequence_len = 512
        tokens = static.data(
            name="tokens", shape=[batch_size, sequence_len], dtype='int64')
        position_ids = static.data(
            name="position_ids",
            shape=[batch_size, sequence_len],
            dtype='int64')
        attention_mask = static.data(
            name="attention_mask",
            shape=[batch_size, 1, sequence_len, sequence_len],
            dtype='float32')
        labels = static.data(
            name="labels", shape=[batch_size, sequence_len], dtype='int64')
        loss_mask = static.data(
            name="loss_mask", shape=[batch_size, sequence_len], dtype='float32')


        data_holders = [tokens, loss_mask, attention_mask, position_ids, labels] 
        train_data_loader, valid_data_loader, test_data_loader = create_data_loader(args, data_holders, topo)

        gpt = GPTModel(
            vocab_size=50304,
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=1024,
            type_vocab_size=1,
            initializer_range=0.02,
            pad_token_id=0,
            eos_token_id=7,
            bos_token_id=0,
            eol_token_id=3,
            topo=topo)

        model = GPTForPretraining(
            gpt, vocab_size=50304, hidden_size=768, initializer_range=0.02)

        preds = model(tokens, position_ids, attention_mask)

        criterion = GPTPretrainingCriterion()

        loss = criterion(preds, labels, loss_mask)

    return train_program, start_program, loss, train_data_loader


def main(args):

    from paddlenlp.ops import Topology
    worker_num = paddle.distributed.get_world_size()
    worker_index = paddle.distributed.get_rank()

    global_setting.init_global()
    global_setting._global_parallel_stratergy = None 
    global_setting._global_process_mesh = None

    topo = Topology(
    device_rank=worker_index,
    world_size=worker_num,
    dp_degree=args.dp_degree,
    pp_degree=1,
    sharding_degree=1,
    mp_degree=args.mp_degree)

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.amp = False
    dist_strategy.pipeline = False
    dist_strategy.recompute = False

    if args.dp_degree > 1 and args.mp_degree == 1:
        # init data parallel optimizer
        dist_strategy.without_graph_optimization = True
    else:
        # init tensor parallel optimizer
        dist_strategy.tensor_parallel = True
        dist_strategy.tensor_parallel_configs = {
            'tensor_parallel_degree': args.mp_degree,
        }
    fleet.init(is_collective=True, strategy=dist_strategy)

    train_program = static.Program()
    start_program = static.Program()
    train_program, start_program, loss, train_data_loader = gpt_pretrain_forward(args, train_program, start_program, topo)

    with static.program_guard(train_program, start_program), utils.unique_name.guard():

        optimizer = paddle.fluid.optimizer.AdamOptimizer(
            learning_rate=0.00001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            grad_clip=None)

        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(loss)
        print(str(dist_strategy))

    with open(args.output_dir + "/hybrid_main_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(train_program))
    with open(args.output_dir + "/hybrid_startup_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(start_program))


    # paddle.seed(worker_index + 2021)
    # random.seed(worker_index + 2021)
    # np.random.seed(worker_index + 2021)

    place = paddle.set_device("gpu")
    exe = paddle.static.Executor(place)
    exe.run(start_program)

    fetchs = [loss] + ['linear_5.w_0']  
    for eval_step, batch in enumerate(train_data_loader):
        loss, param = exe.run(train_program, feed=batch, fetch_list=fetchs)
        print("step: %d, loss: %f" % (eval_step, loss[0]))


def print_param(program):
    from paddle.fluid.framework import Parameter
    def is_parameter(var):
        return isinstance(var, Parameter)
    def get_tensor(var):
        t = paddle.fluid.global_scope().find_var(var.name).get_tensor()
        return np.array(t)
    def get_name(var):
        return var.name
    parameter_list = list(filter(is_parameter, program.list_vars()))
    for p in sorted(parameter_list, key=get_name):
        print(p.name)
        print(get_tensor(p)[:20])



def get_train_data_file(path):
    files = [
        os.path.join(path, f) for f in os.listdir(path)
        if (os.path.isfile(os.path.join(path, f)) and "npz_" not in
            str(f))
    ]

    data_file = files[0]
    return data_file

def create_data_loader(args, data_holders, topo):

    from dataset import create_pretrained_dataset
    from paddlenlp.transformers import GPTTokenizer, GPTChineseTokenizer

    data_file = get_train_data_file("./data")

    tokenizer_class = GPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path)
    eod_id = tokenizer.eod_token_id
    train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
        args,
        data_file,
        data_world_size=topo.data_info.size,
        data_world_rank=topo.data_info.rank,
        eos_id=eod_id,
        max_seq_len=args.max_seq_len,
        places=paddle.static.cuda_places(),
        data_holders=data_holders,
        pipeline_mode=False)
    return train_data_loader, valid_data_loader, test_data_loader 


if __name__ == "__main__":
    import shutil
    import os
    config = parse_args()
    main(config)
