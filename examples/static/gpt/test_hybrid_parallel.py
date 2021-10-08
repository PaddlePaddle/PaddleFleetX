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
# import paddlenlp
import modeling_utils
import re

paddle.enable_static()

def save_checkpoint(exe, save_path, train_program, num_sharding=1, num_pp=1):
    if num_sharding > 1:
        fleet.meta_optimizers.sharding.utils.save_persistables(exe, save_path, train_program)
    elif num_pp > 1:
        fluid.io.save_persistables(exe, save_path, train_program._pipeline_opt['section_program'])
    else:
        fluid.io.save_persistables(exe, save_path, train_program)

def init_checkpoint(exe, init_checkpoint_path, main_program, train_program=None):
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        if not fluid.io.is_persistable(var):
            return False
        # is_distributed role to rule out mp params
        # block = paddle.static.default_main_program().global_block()
        if train_program:
            block = train_program.global_block()
            re_str1 = "_pow_acc_0"
            re_str2 = "_moment"
            var_name = var.name
            if re_str1 in var.name:
                var_name = "_".join(var.name.split(re_str1)[0].split("_")[:-1])
            if re_str2 in var.name:
                var_name = var.name.split(re_str2)[0]
            # pipeline may change the variable name, the variable belongs to grad
            if block.has_var(var_name):
                var_master = block.var(var_name)
                if fluid.io.is_parameter(var_master) and var_master.is_distributed:
                    return False
                else:
                    return os.path.exists(os.path.join(init_checkpoint_path, var.name))
            else:
                return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    print("Load model from {}".format(init_checkpoint_path))

def checkpoint_rearrange(save_model_dir, init_checkpoint_path, index, num_pp, num_mp, steps, dp_rank):
    matchObj = re.search(r'.*saved_model_pp(\d+)mp(\d+).*', init_checkpoint_path, re.M|re.I)
    if matchObj:
        pre_num_pp = int(matchObj.group(1))
        pre_num_mp = int(matchObj.group(2))
    else:
        raise Exception('saved_model_pp(\d+)mp(\d+) must in init_checkpoint_path')

    if pre_num_pp < num_pp:
        raise Exception('current num_pp({}) must less than or equal to checkpoint num_pp({})'.format(num_pp, pre_num_pp))

    if pre_num_mp < num_mp:
        raise Exception('current num_mp({}) must equal to checkpoint num_mp({})'.format(num_mp, pre_num_mp))

    pre_topo = Topology(rank=index, world_size=pre_num_pp*pre_num_mp, dp=1, pp=pre_num_pp, sharding=1, mp=pre_num_mp)
    cur_topo = Topology(rank=index, world_size=num_pp*num_mp, dp=1, pp=num_pp, sharding=1, mp=num_mp)
    start_index = cur_topo.pp.world.index(index) * len(cur_topo.pp.world)
    pre_ranks = pre_topo.pp.world[start_index:start_index+len(cur_topo.pp.world)]
    dst_dir = os.path.join(save_model_dir, 'rank_' + str(index), 'step_' + str(steps))
    lock_file_name = os.path.join(save_model_dir, 'rank_' + str(index), 'step_' + str(steps) + '.lock')
    if dp_rank == 0:
        shutil.rmtree(dst_dir, ignore_errors=True)
        os.makedirs(dst_dir)
        for idx in pre_ranks:
            src_dir = os.path.join(init_checkpoint_path, 'rank_' + str(idx), 'step_' + str(steps))
            for dirpath, dirnames, filenames in os.walk(src_dir):
                for filename in filenames:
                    src_path = os.path.join(dirpath, filename)
                    shutil.copy(src_path, dst_dir)
        with open(lock_file_name, 'w') as f:
            pass
        time.sleep(3)
        os.remove(lock_file_name)
    else:
        while True:
            if not os.path.exists(lock_file_name):
                time.sleep(1)
            else:
                break



def gpt_pretrain_forward(args, train_program, start_program, topo):
    from modeling import GPTModel, GPTForPretraining, GPTPretrainingCriterion
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        sequence_len = 512
        tokens = static.data(
            name="tokens", shape=[-1, sequence_len], dtype='int64')
        position_ids = static.data(
            name="position_ids",
            shape=[-1, sequence_len],
            dtype='int64')
        attention_mask = static.data(
            name="attention_mask",
            shape=[-1, 1, sequence_len, sequence_len],
            dtype='float32')
        labels = static.data(
            name="labels", shape=[-1, sequence_len], dtype='int64')
        loss_mask = static.data(
            name="loss_mask", shape=[-1, sequence_len], dtype='float32')
        with paddle.static.device_guard("gpu:0"):
            data_holders = [tokens, loss_mask, attention_mask, position_ids, labels] 
            train_data_loader, valid_data_loader, test_data_loader = create_data_loader(args, data_holders, topo)
    
        gpt = modeling_utils.ops.guard(f'gpu:0')(GPTModel)(
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
            topo=topo,
            debug=args.debug)

        model = modeling_utils.ops.guard(f'gpu:{args.pp_degree-1}')(GPTForPretraining)(
            gpt, vocab_size=50304, hidden_size=768, initializer_range=0.02)

        preds = model(tokens, position_ids, attention_mask)

        criterion = modeling_utils.ops.guard(f'gpu:{args.pp_degree-1}')(GPTPretrainingCriterion)(args, topo)

        loss_vars = criterion(preds, labels, loss_mask)

        if topo.pp_info.size > 1:
            pp_loss = loss_vars["pp_total_loss"]
            pp_loss_name = pp_loss.name
            for op in train_program.global_block().ops:
                if op.type == "fill_constant" and op.desc.output_arg_names()[0] == pp_loss_name:
                    op._set_attr("op_role", 16)
                    op._set_attr("op_device", f"gpu:%d" % (args.pp_degree-1))
        
    return train_program, start_program, loss_vars, train_data_loader

def dist_optimizer(args, topo):
    default_global_batch_size = topo.data_info.size * args.micro_batch_size
    if args.global_batch_size is None:
        args.global_batch_size = default_global_batch_size

    bsz_per_dp = args.global_batch_size // topo.data_info.size
    micro_batch_size = args.micro_batch_size
    assert args.global_batch_size % micro_batch_size == 0, f"cannot do gradient accumulate, global_batch_size: {args.global_batch_size} micro_batch_size: {micro_batch_size}"
    acc_steps = bsz_per_dp // micro_batch_size

    exec_strategy = paddle.fluid.ExecutionStrategy()
    exec_strategy.num_threads = 2
    exec_strategy.num_iteration_per_drop_scope = 1

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.nccl_comm_num = 3

    dist_strategy.recompute = args.use_recompute
    dist_strategy.pipeline = args.pp_degree > 1

    if args.use_amp:
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "custom_white_list": ['softmax', 'layer_norm', 'gelu', 'fused_elemwise_activation'],
            "init_loss_scaling": 32768,
            "use_dynamic_loss_scaling": True,
        }

    if args.mp_degree > 1 or args.pp_degree > 1:
        if args.mp_degree > 1 and \
                args.pp_degree == 1 and args.sharding_degree == 1:
            dist_strategy.tensor_parallel = True
            dist_strategy.tensor_parallel_configs = {"tensor_parallel_degree": args.mp_degree}
        else:
            print("pp sharding")
            dist_strategy.sharding = True
            dist_strategy.sharding_configs = {
                "segment_broadcast_MB": 32,
                "sharding_degree": args.sharding_degree,
                "mp_degree": args.mp_degree,
                "pp_degree": args.pp_degree,
                "dp_degree": args.dp_degree,
                "optimize_offload": False,
            }
    else:
        if args.dp_degree > 1:
            dist_strategy.without_graph_optimization = True

    if args.pp_degree > 1:

        dist_strategy.pipeline_configs = {
            "schedule_mode": "F-then-B",
            "micro_batch_size": micro_batch_size,
            "accumulate_steps": acc_steps
        }
    else:
        assert acc_steps == 1, f"Only support accumulate steps in piplinemode. Please set you global_batch_size={default_global_batch_size}"

    return dist_strategy


def main(args):

    from modeling_utils.ops import Topology
    worker_num = paddle.distributed.get_world_size()
    worker_index = paddle.distributed.get_rank()

    topo = Topology(
    device_rank=worker_index,
    world_size=worker_num,
    dp_degree=args.dp_degree,
    pp_degree=args.pp_degree,
    sharding_degree=1,
    mp_degree=args.mp_degree)
    print(topo)

    dist_strategy = dist_optimizer(args, topo)
    fleet.init(is_collective=True, strategy=dist_strategy)
    
    train_program = static.Program()
    start_program = static.Program()
    train_program, start_program, loss_vars, train_data_loader = gpt_pretrain_forward(args, train_program, start_program, topo)
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        optimizer = paddle.fluid.optimizer.AdamOptimizer(
            learning_rate=0.00001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            grad_clip=None)

        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(loss_vars["total_loss"])
        print(str(fleet._final_strategy()))
    print("The training meta optimizer is/are %s" %
                        fleet._get_applied_meta_list())
    with open(args.output_dir + "/hybrid_main_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(train_program))
    with open(args.output_dir + "/hybrid_startup_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(start_program))

    if args.debug:
        args.seed = 2021
        paddle.seed(2021)
        random.seed(2021)
        np.random.seed(2021)
    else:
        paddle.seed(worker_index + 2021)
        random.seed(worker_index + 2021)
        np.random.seed(worker_index + 2021)
    place = paddle.set_device("gpu")
    exe = paddle.static.Executor(place)
    exe.run(start_program)
    is_single = (args.dp_degree == 1 and args.mp_degree == 1 and args.pp_degree == 1)
    if args.debug and (args.pp_degree > 1 or is_single):
        checkpoint_path = args.checkpoint_path
        if train_program._pipeline_opt:
            with open(args.output_dir + "/section_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
                f.write(str(train_program._pipeline_opt['section_program']))
            init_checkpoint(exe, checkpoint_path, \
                            train_program._pipeline_opt['section_program'], train_program)
            
        else:
            init_checkpoint(exe, checkpoint_path, train_program)

    step = 0
    while True:
        train_data_loader.start()
        while True:
            if args.pp_degree == 1:
                fetchs = [loss_vars["total_loss"]]
            else:
                fetchs = [loss_vars["pp_total_loss"]]
            ret = exe.run(train_program, fetch_list=fetchs)
            print("step: ", step, "loss_print: ", ret)
            step += 1
            if step >= 20000:
                break
        train_data_loader.reset()
        break




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
    from modeling_utils.transformers import GPTTokenizer, GPTChineseTokenizer

    data_file = get_train_data_file("./data")

    tokenizer_class = GPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path)
    eod_id = tokenizer.eod_token_id
    train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
        args,
        data_file,
        topo,
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


