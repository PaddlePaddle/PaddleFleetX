import os
import six
import paddle.fluid as fluid
from paddle.fluid.framework import Program
from paddle.fluid.framework import Block
from paddle.fluid import core

class FleetProgram(Program):
    def __init__(self):
        super(FleetProgram, self).__init__()

    def _set_input_names(self, input_names):
        self.input_names = input_names

    def _set_loss_name(self, loss_name):
        self.loss_name = loss_name

    def _set_param_names(self, param_names):
        self.param_names = param_names

    def _set_hidden_var_names(self, hidden_var_names):
        self.hidden_names = hidden_var_names

    def parse_from_string(self, binary_str):
        p = FleetProgram()
        p.desc = core.ProgramDesc(binary_str)
        p.blocks = [Block(p, i) for i in six.moves.range(p.desc.num_blocks())]
        p._sync_with_cpp()
        return p

    def all_parameters(self):
        parameters = []
        for each_block in self.blocks:
            for item in six.iteritems(each_block.vars):
                if item[0] in self.param_names:
                    parameters.append(item[1])
        return parameters
        

def load_program(program_input):
    with open(program_input + '/startup_program', "rb") as fin:
        program_desc_str = fin.read()
        new_startup = FleetProgram().parse_from_string(program_desc_str)

    with open(program_input + '/main_program', "rb") as fin:
        program_desc_str = fin.read()
        new_main = FleetProgram().parse_from_string(program_desc_str)

    para_list = []
    with open(program_input + '/para_info', 'r') as fin:
        for line in fin:
            current_para = line[:-1]
            para_list.append(current_para)
    new_startup._set_param_names(para_list)
    new_main._set_param_names(para_list)

    input_list = []
    with open(program_input + '/input_names', 'r') as fin:
        for line in fin:
            current_input = line[:-1].split(":")[0].replace("var", "").strip()
            input_list.append(current_input)
    new_startup._set_input_names(input_list)
    new_main._set_input_names(input_list)

    fluid.framework.switch_main_program(new_main)
    fluid.framework.switch_startup_program(new_startup)
    with open(program_input + '/loss_name', 'r') as fin:
        loss_name = fin.read()
    new_startup._set_loss_name(loss_name)
    new_main._set_loss_name(loss_name)

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
        main_para.regularizer = None
        main_para.optimize_attr = {'learning_rate': 1.0}
        main_para.trainable = True
        main_para.is_distributed = False
 
        startup_para = new_startup.global_block().var(item)
        startup_para.regularizer = None
        startup_para.optimize_attr = {'learning_rate': 1.0}
        startup_para.trainable = True
        startup_para.is_distributed = False

    exe = fluid.Executor(fluid.CPUPlace())
    loss = None
    input_vars = []
    print(input_list)
    for var in new_main.list_vars():
        if var.name in input_list:
            input_vars.append(var)
        if var.name == loss_name:
            loss = var
        if lr_name != None:
            if var.name == lr_name:
                lr = var

    return input_vars, loss, new_startup, new_main, unique_generator


def save_program(main_prog,
                 startup_prog,
                 program_path,
                 input_list,
                 hidden_vars,
                 loss,
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
        for id,value in generator_info.iteritems():
            fout.write("%s:%s\n" % (id,value))
