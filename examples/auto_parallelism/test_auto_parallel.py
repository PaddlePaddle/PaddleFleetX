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
import paddle.distributed.auto_parallel as auto
import paddle.static as static
from paddle.distributed import fleet
from args import parse_args
import modeling_utils
import global_setting

paddle.enable_static()


def gpt_pretrain_forward(args, train_program, start_program, topo):
    from modeling import GPTModel, GPTForPretraining, GPTPretrainingCriterion
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        batch_size = args.global_batch_size 
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

        if global_setting._global_parallel_stratergy == "dp":
            auto.shard_tensor(
                tokens, global_setting._global_process_mesh, dim_mapping=[0, -1])
        elif global_setting._global_parallel_stratergy == "dp_mp":
            auto.shard_tensor(
                tokens, global_setting._global_process_mesh, dim_mapping=[0, -1])
        elif global_setting._global_parallel_stratergy == "pp":
            auto.shard_tensor(
                tokens, global_setting.PP_MESH_LIST[0], dim_mapping=[-1, -1])
            auto.shard_tensor(
                attention_mask, global_setting.PP_MESH_LIST[0], dim_mapping=[-1, -1, -1, -1])
        elif global_setting._global_parallel_stratergy == "dp_pp":
            auto.shard_tensor(
                tokens, global_setting.DPPP_MESH_LIST[0], dim_mapping=[0, -1])
        elif global_setting._global_parallel_stratergy == "mp_pp":
            auto.shard_tensor(tokens, global_setting.MPPP_MESH_LIST[0], dim_mapping=[-1, -1])
        elif global_setting._global_parallel_stratergy == "dp_mp_pp":
            auto.shard_tensor(
                tokens, global_setting.DPMPPP_MESH_LIST[0], dim_mapping=[0, -1])

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
            # NOTE None topo to got serial program
            topo=None,
            debug=args.debug,
            pp_degree=args.pp_degree)

        model = GPTForPretraining(
            gpt, vocab_size=50304, hidden_size=768, initializer_range=0.02)

        preds = model(tokens, position_ids, attention_mask)

        criterion = GPTPretrainingCriterion(args=args, topo=topo)

        loss_vars = criterion(preds, labels, loss_mask)

    return train_program, start_program, loss_vars, train_data_loader


def main(args):

    from modeling_utils.ops import Topology
    worker_num = paddle.distributed.get_world_size()
    worker_index = paddle.distributed.get_rank()

    global_setting.init_global()
    if args.dp_degree == 1 and args.mp_degree == 1 and args.pp_degree == 1:
        set_global_parallel_strategy("None")
        set_global_process_mesh("None")
    elif args.dp_degree > 1 and args.mp_degree == 1 and args.pp_degree == 1:
        global_setting._global_parallel_stratergy = "dp"
        ROOT_MESH = auto.ProcessMesh([0, 1])
        global_setting._global_process_mesh = auto.ProcessMesh(
            mesh=[0, 1], parent=ROOT_MESH)
    elif args.dp_degree == 1 and args.mp_degree > 1 and args.pp_degree == 1:
        global_setting._global_parallel_stratergy = "mp"
        ROOT_MESH = auto.ProcessMesh([0, 1])
        global_setting._global_process_mesh = auto.ProcessMesh(
            mesh=[0, 1], parent=ROOT_MESH)
    elif args.dp_degree > 1 and args.mp_degree > 1 and args.pp_degree == 1:
        global_setting._global_parallel_stratergy = "dp_mp"
        ROOT_MESH = auto.ProcessMesh([[0, 1], [2, 3]])
        global_setting._global_process_mesh = auto.ProcessMesh(
            mesh=[[0, 1], [2, 3]], parent=ROOT_MESH)
    elif args.dp_degree == 1 and args.mp_degree == 1 and args.pp_degree > 1:
        global_setting._global_parallel_stratergy = "pp"
        ROOT_MESH = auto.ProcessMesh([0, 1])
        global_setting._global_process_mesh = auto.ProcessMesh(mesh=[0, 1], parent=ROOT_MESH)
        global_setting.PP_MESH_LIST = [auto.ProcessMesh(mesh=[0], parent=ROOT_MESH), auto.ProcessMesh(mesh=[1], parent=ROOT_MESH)]
    elif args.pp_degree > 1 and args.dp_degree > 1 and args.mp_degree == 1:
        global_setting._global_parallel_stratergy = "dp_pp"
        ROOT_MESH = auto.ProcessMesh([[0, 1], [2, 3]])
        global_setting._global_process_mesh = auto.ProcessMesh(mesh=[[0, 2], [1, 3]], parent=ROOT_MESH)
        global_setting.DPPP_MESH_LIST = [auto.ProcessMesh(mesh=[0, 2], parent=ROOT_MESH), auto.ProcessMesh(mesh=[1, 3], parent=ROOT_MESH)]
    elif args.mp_degree > 1 and args.pp_degree > 1 and args.dp_degree == 1:
        global_setting._global_parallel_stratergy = "mp_pp"
        ROOT_MESH = auto.ProcessMesh([[0, 1], [2, 3]])
        global_setting._global_process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]], parent=ROOT_MESH)
        global_setting.MPPP_MESH_LIST = [auto.ProcessMesh(mesh=[0, 1], parent=ROOT_MESH), auto.ProcessMesh(mesh=[2, 3], parent=ROOT_MESH)]
    elif args.mp_degree > 1 and args.pp_degree > 1 and args.dp_degree > 1:
        global_setting._global_parallel_stratergy = "dp_mp_pp"
        ROOT_MESH = auto.ProcessMesh([[[0, 1], [4, 5]], [[2, 3], [6, 7]]])
        global_setting._global_process_mesh = auto.ProcessMesh(mesh=[[[0, 1], [4, 5]], [[2, 3], [6, 7]]], parent=ROOT_MESH)
        global_setting.DPMPPP_MESH_LIST = [auto.ProcessMesh(mesh=[[0, 1], [4, 5]], parent=ROOT_MESH), auto.ProcessMesh(mesh=[[2, 3], [6, 7]], parent=ROOT_MESH)]
        
    topo = Topology(
    device_rank=worker_index,
    world_size=worker_num,
    dp_degree=args.dp_degree,
    pp_degree=args.pp_degree,
    sharding_degree=1,
    mp_degree=args.mp_degree)

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.amp = False
    dist_strategy.pipeline = False
    dist_strategy.recompute = False

    # init parallel optimizer
    dist_strategy.semi_auto = True
    fleet.init(is_collective=True, strategy=dist_strategy)

    train_program = static.Program()
    start_program = static.Program()
    train_program, start_program, loss_vars, train_data_loader = gpt_pretrain_forward(args, train_program, start_program, topo)

    # different from hybrid parallel
    optimizer = paddle.fluid.optimizer.AdamOptimizer(
        learning_rate=0.00001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        grad_clip=None)

    optimizer = fleet.distributed_optimizer(optimizer)
    _, _, distributed_startup_program, distributed_main_program = optimizer.minimize(loss_vars["total_loss"], start_program)
    print(str(dist_strategy))

    with open(args.output_dir + "/serial_main_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(train_program))
    if "pp" not in global_setting._global_parallel_stratergy:   
        with open(args.output_dir + "/serial_startup_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
            f.write(str(start_program))

    with open(args.output_dir + "/auto_main_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(distributed_main_program))
    with open(args.output_dir + "/auto_startup_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(distributed_startup_program))

    gen = paddle.seed(worker_index + 2021)
    random.seed(worker_index + 2021)
    np.random.seed(worker_index + 2021)

    place = paddle.set_device("gpu")
    exe = paddle.static.Executor(place)
   
    exe.run(distributed_startup_program)

    # for dp or mp, in this demo, dp_degree or mp_degree is 2
    if args.pp_degree < 2:
        while True:
            train_data_loader.start()
            eval_step = 0
            while True:
                fetchs = [loss_vars["total_loss"]]
                loss_print = exe.run(distributed_main_program, fetch_list=fetchs)
                print("step: %d, loss_print: %f" % (eval_step, loss_print[0]))
                eval_step += 1
                if eval_step >= 20000:
                    break
            train_data_loader.reset()
            break
    
    # for pp, in this demo, pp_degree is 2
    if args.pp_degree > 1 and args.dp_degree == 1 and args.mp_degree == 1:
        while True:
            train_data_loader.start()
            eval_step = 0
            while True:
                fetchs = [loss_vars["total_loss"]]
                if paddle.distributed.get_rank() in [0]:
                    res = exe.run(distributed_main_program)
                    print("step: %d, loss_print: %s" % (eval_step, "None"))
                else:
                    loss_print = exe.run(distributed_main_program, fetch_list=fetchs)
                    print("step: %d, loss_print: %f" % (eval_step, loss_print[0]))
                eval_step += 1
                if eval_step >= 20000:
                    break
            train_data_loader.reset()
            break

    # for dp+pp, in this demo, dp_degree is 2 and pp_degree is 2
    if args.pp_degree > 1 and args.dp_degree > 1 and args.mp_degree == 1:
        while True:
            train_data_loader.start()
            eval_step = 0
            while True:
                fetchs = [loss_vars["total_loss"]]
                if paddle.distributed.get_rank() in [0, 2]:
                    res = exe.run(distributed_main_program)
                    print("step: %d, loss_print: %s" % (eval_step, "None"))
                else:
                    loss_print = exe.run(distributed_main_program, fetch_list=fetchs)
                    print("step: %d, loss_print: %f" % (eval_step, loss_print[0]))
                eval_step += 1
                if eval_step >= 20000:
                    break
            train_data_loader.reset()
            break

    # for mp+pp, in this demo, mp_degree is 2 and pp_degree is 2
    if args.pp_degree > 1 and args.mp_degree > 1 and args.dp_degree == 1:
        while True:
            train_data_loader.start()
            eval_step = 0
            while True:
                fetchs = [loss_vars["total_loss"]]
                if paddle.distributed.get_rank() in [0, 1]:
                    exe.run(distributed_main_program)
                    print("step: %d, loss_print: %s" % (eval_step, "None"))
                else:
                    loss_print = exe.run(distributed_main_program, fetch_list=fetchs)
                    print("step: %d, loss_print: %f" % (eval_step, loss_print[0]))
                eval_step += 1
                if eval_step >= 20000:
                    break
            train_data_loader.reset()
            break

    # for dp+mp+pp, in this demo, dp_degree is 2 and mp_degree is 2 and pp_degree is 2
    if args.pp_degree > 1 and args.mp_degree > 1 and args.dp_degree > 1:
        while True:
            train_data_loader.start()
            eval_step = 0
            while True:
                if paddle.distributed.get_rank() in [0, 1, 4, 5]:
                    exe.run(distributed_main_program)
                    print("step: %d, loss_print: %s" % (eval_step, "None"))
                else:
                    fetchs = [loss_vars["total_loss"]]
                    loss_print = exe.run(distributed_main_program, fetch_list=fetchs)
                    print("step: %d, loss_print: %f" % (eval_step, loss_print[0]))
                eval_step += 1
                if eval_step >= 20000:
                    break
            train_data_loader.reset()
            break


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

    from modeling_utils.transformers import GPTTokenizer, GPTChineseTokenizer

    data_file = get_train_data_file("./data")

    tokenizer_class = GPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path)
    eod_id = tokenizer.eod_token_id
    train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
        args,
        data_file,
        topo=topo,
        eod_id=eod_id,
        max_seq_len=args.max_seq_len,
        places=paddle.static.cuda_places(),
        data_holders=data_holders,
        pipeline_mode=True)
    return train_data_loader, valid_data_loader, test_data_loader 


if __name__ == "__main__":
    import shutil
    import os
    config = parse_args()
    main(config)


