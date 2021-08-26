#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import six
import ast
import copy
import re
import shutil
import time

import numpy as np
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import paddle
from .topo import Topology

def init_pretraining_params(exe,
                            pretraining_params_path,
                            main_program,
                            ):
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        var_path = os.path.join(pretraining_params_path, var.name)
        var_exists = os.path.exists(var_path)
        if not isinstance(var, fluid.framework.Parameter):
            return False
        else:
            if var_exists:
                print('loading {} for {}'.format(var_path, var.name))
            else:
                print('do not detecting param {} for {}'.format(var.name, var_path))
        return var_exists

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=existed_params)
    print("Load pretraining parameters from {}.".format(pretraining_params_path))


def init_checkpoint(exe, init_checkpoint_path, main_program):
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        if not fluid.io.is_persistable(var):
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

def save_checkpoint(exe, save_path, train_program, num_sharding=1, num_pp=1):
    if num_sharding > 1:
        fleet.meta_optimizers.sharding.utils.save_persistables(exe, save_path, train_program)
    elif num_pp > 1:
        fluid.io.save_persistables(exe, save_path, train_program._pipeline_opt['section_program'])
    else: 
        fluid.io.save_persistables(exe, save_path, train_program)

def save_inference_model(dirname,
                         feeded_var_names,
                         fetch_var_names,
                         exe,
                         train_program,
                         num_sharding=1,
                         num_pp=1,
                         is_first=False,
                         is_last=False):
    if num_sharding > 1:
        raise Exception('num_sharding > 1 do not supported save_inference_model now.')
    elif num_pp > 1:
        op_role_key = fluid.core.op_proto_and_checker_maker.kOpRoleAttrName()
        op_role_forward = int(fluid.core.op_proto_and_checker_maker.OpRole.Forward)

        ops = train_program._pipeline_opt['section_program'].global_block().ops
        if is_first:
            fetch_var_names = []
            program_input_names = []
            last_idx = -1
            for idx, op in enumerate(ops):
                if (op.type == 'partial_send' or op.type == 'send_v2') and op.attr(op_role_key) == op_role_forward:
                    fetch_var_names += op.input("X")
                    last_idx = max(last_idx, idx)
                elif op.type == 'read' and op.attr(op_role_key) == op_role_forward:
                    program_input_names += op.output("Out")
                    
            for idx, op in enumerate(ops):
                if idx > last_idx:
                    break
                for input_name in program_input_names:
                    if input_name in op.input_arg_names and input_name not in feeded_var_names:
                        feeded_var_names.append(input_name)
                
        elif is_last:
            feeded_var_names = []
            program_input_names = []
            last_idx = -1
            for idx, op in enumerate(ops):
                if (op.type == 'partial_recv' or op.type == 'recv_v2') and op.attr(op_role_key) == op_role_forward:
                    feeded_var_names += op.output("Out")
                elif op.type == 'read' and op.attr(op_role_key) == op_role_forward:
                    program_input_names += op.output("Out")
                    
                for output_name in fetch_var_names:
                    if output_name in op.output_arg_names:
                        last_idx = max(last_idx, idx)
                        
            for idx, op in enumerate(ops):
                if idx > last_idx:
                    break
                for input_name in program_input_names:
                    if input_name in op.input_arg_names and input_name not in feeded_var_names:
                        feeded_var_names.append(input_name)
                        
        else:
            feeded_var_names = []
            fetch_var_names = []
            program_input_names = []
            last_idx = -1
            for idx, op in enumerate(ops):
                if (op.type == 'partial_recv' or op.type == 'recv_v2') and op.attr(op_role_key) == op_role_forward:
                    feeded_var_names += op.output("Out")
                elif (op.type == 'partial_send' or op.type == 'send_v2') and op.attr(op_role_key) == op_role_forward:
                    fetch_var_names += op.input("X")
                    last_idx = max(last_idx, idx)
                elif op.type == 'read' and op.attr(op_role_key) == op_role_forward:
                    program_input_names += op.output("Out")
                    
            for idx, op in enumerate(ops):
                if idx > last_idx:
                    break
                for input_name in program_input_names:
                    if input_name in op.input_arg_names and input_name not in feeded_var_names:
                        feeded_var_names.append(input_name)

        feeded_vars = [train_program.global_block().vars[var_name] for var_name in feeded_var_names]
        fetch_vars = [train_program.global_block().vars[var_name] for var_name in fetch_var_names]
                    
        paddle.static.save_inference_model(dirname, feeded_vars, fetch_vars, exe,
            program=train_program._pipeline_opt['section_program'])
    else: 
        feeded_vars = [train_program.global_block().vars[var_name] for var_name in feeded_var_names]
        fetch_vars = [train_program.global_block().vars[var_name] for var_name in fetch_var_names]
        paddle.static.save_inference_model(dirname, feeded_vars, fetch_vars, exe,
            program=train_program)
