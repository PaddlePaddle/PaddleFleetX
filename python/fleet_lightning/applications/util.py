# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import six
import paddle.fluid as fluid
from paddle.fluid.framework import Program, Parameter
from paddle.fluid import core


def load_program(program_input):
    with open(program_input + '/startup_program', "rb") as fin:
        program_desc_str = fin.read()
        new_startup = Program().parse_from_string(program_desc_str)

    with open(program_input + '/main_program', "rb") as fin:
        program_desc_str = fin.read()
        new_main = Program().parse_from_string(program_desc_str)

    para_list = []
    with open(program_input + '/para_info', 'r') as fin:
        for line in fin:
            current_para = line[:-1]
            para_list.append(current_para)

    input_list = []
    with open(program_input + '/input_names', 'r') as fin:
        for line in fin:
            current_input = line[:-1].split(":")[0].replace("var", "").strip()
            input_list.append(current_input)

    with open(program_input + '/loss_name', 'r') as fin:
        loss_name = fin.read()

    unique_generator = fluid.unique_name.UniqueNameGenerator()
    with open(program_input + '/unique_name_guard', 'r') as fin:
        for line in fin:
            current_guard = line[:-1].split(":")
            unique_generator.ids[current_guard[0]] = int(current_guard[1])

    fluid.unique_name.switch(unique_generator)
    if os.path.exists(program_input + '/lr_name'):
        with open(program_input + '/lr_name', 'r') as fin:
            lr_name = fin.read()
    else:
        lr_name = None

    for item in para_list:
        main_para = new_main.global_block().var(item)
        main_para.__class__ = Parameter
        main_para.regularizer = None
        main_para.optimize_attr = {'learning_rate': 1.0}
        main_para.trainable = True
        main_para.is_distributed = False

        startup_para = new_startup.global_block().var(item)
        startup_para.__class__ = Parameter
        startup_para.regularizer = None
        startup_para.optimize_attr = {'learning_rate': 1.0}
        startup_para.trainable = True
        startup_para.is_distributed = False

    exe = fluid.Executor(fluid.CPUPlace())
    loss = None
    input_vars = []
    for input_name in input_list:
        for var in new_main.list_vars():
            if var.name == input_name:
                input_vars.append(var)
    for var in new_main.list_vars():
        if var.name == loss_name:
            loss = var
        if lr_name != None:
            if var.name == lr_name:
                lr = var

    fluid.framework.switch_main_program(new_main)
    fluid.framework.switch_startup_program(new_startup)

    return input_vars, loss, new_startup, new_main, unique_generator


def save_program(main_prog,
                 startup_prog,
                 program_path,
                 input_list,
                 hidden_vars,
                 loss,
                 generator_info,
                 learning_rate=None):
    if not os.path.exists(program_path):
        os.makedirs(program_path)
    main_program_str = main_prog.desc.serialize_to_string()
    startup_program_str = startup_prog.desc.serialize_to_string()
    params = main_prog.global_block().all_parameters()
    para_info = []
    for pa in params:
        para_info.append(pa.name)
    with open(program_path + '/input_names', 'w') as fout:
        for input in input_list:
            fout.write("%s\n" % input)
    if hidden_vars != None:
        with open(program_path + '/hidden_vars', 'w') as fout:
            for var in hidden_vars:
                fout.write("%s:%s\n" % (var[0], var[1].name))
    with open(program_path + '/para_info', 'w') as fout:
        for item in para_info:
            fout.write("%s\n" % item)
    with open(program_path + '/startup_program', "wb") as fout:
        fout.write(startup_program_str)
    with open(program_path + '/main_program', "wb") as fout:
        fout.write(main_program_str)
    with open(program_path + '/loss_name', 'w') as fout:
        fout.write(loss.name)
    if type(learning_rate) == fluid.Variable:
        with open(program_path + '/lr_name', 'w') as fout:
            fout.write(learning_rate.name)
    with open(program_path + '/unique_name_guard', 'w') as fout:
        for id, value in generator_info.iteritems():
            fout.write("%s:%s\n" % (id, value))
