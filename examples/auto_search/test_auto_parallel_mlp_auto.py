# # Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import random
# import numpy as np
# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F
# import paddle.utils as utils
# import paddle.static as static
# import paddle.distributed.auto_parallel as auto

# from paddle.distributed import fleet
# from paddle.fluid.initializer import NumpyArrayInitializer
# from paddle.distributed.auto_parallel.utils import save_distributed_checkpoint, load_distributed_checkpoint
# from paddle.distributed.auto_parallel.utils import get_dist_attr, merge_and_slice_parameter, load_parameter_into_program
# from paddle.distributed.auto_parallel.utils import load_checkpoint_into_program
# from paddle.distributed.auto_parallel.dist_context import get_default_distributed_context

# import global_setting

# paddle.enable_static()

# _global_parallel_strategy = None
# _global_process_mesh = None
# PP_MESH_0 = None
# PP_MESH_1 = None


# class MLPLayer(nn.Layer):
#     def __init__(self,
#                  d_model,
#                  dim_feedforward,
#                  initializer_range=0.02,
#                  mesh_idx=None):
#         super(MLPLayer, self).__init__()
#         arr0 = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
#         arr1 = np.random.normal(0, 0.02, size=(dim_feedforward, d_model))
#         weight_attr0 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr0))
#         weight_attr1 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr1))
#         self.linear0 = nn.Linear(
#             d_model,
#             dim_feedforward,
#             weight_attr=weight_attr0,
#             bias_attr=None
#         )
#         self.linear1 = nn.Linear(
#             dim_feedforward,
#             d_model,
#             weight_attr=weight_attr1,
#             bias_attr=None
#         )
#         self.norm = nn.LayerNorm(d_model, epsilon=1e-5)

#     def forward(self, inputs):
#         out = self.norm(inputs)
#         out = self.linear0(out)
#         out = F.gelu(out, approximate=True)
#         out = self.linear1(out)
#         return out


# class MLPModel(nn.Layer):
#     def __init__(self,
#                  input_dim=512,
#                  hidden_size=768,
#                  num_hidden_layers=5,
#                  intermediate_size=4 * 768,
#                  initializer_range=0.02):
#         super(MLPModel, self).__init__()

#         arr = np.random.normal(0, 0.02, size=(input_dim, hidden_size))
#         weight_attr = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr))
#         self.linear0 = nn.Linear(
#             input_dim, 
#             hidden_size,
#             weight_attr=weight_attr,
#             bias_attr=None
#         )
#         self.linear1 = nn.Linear(
#             hidden_size,
#             input_dim, 
#             weight_attr=weight_attr,
#             bias_attr=None
#         )

#         self.mlp_layers = nn.LayerList()
#         for _ in range(num_hidden_layers):
#             mesh_idx = None
#             self.mlp_layers.append(
#                 MLPLayer(
#                     d_model=hidden_size,
#                     dim_feedforward=intermediate_size,
#                     initializer_range=initializer_range,
#                     mesh_idx=mesh_idx
#                 )
#             )
#         self.norm = nn.LayerNorm(input_dim)

#     def forward(self, inputs):
#         output = self.linear0(inputs)
#         for _, layer in enumerate(self.mlp_layers):
#             output = layer(output)
#         output = self.norm(self.linear1(output))
#         return output


# def mlp_forward(train_program, start_program):
#     with static.program_guard(train_program, start_program), utils.unique_name.guard():
#         batch_size = 4
#         input_dim = 512
#         tokens = static.data(
#             name="tokens", shape=[batch_size, input_dim], dtype='float32')
#         labels = static.data(
#             name="labels", shape=[batch_size, input_dim], dtype='float32')

#         if _global_parallel_strategy == "dp":
#             auto.shard_tensor(
#                 tokens,
#                 dist_attr={
#                     "process_mesh": _global_process_mesh,
#                     "dims_mapping": [0, -1]
#                 })

#         mlp = MLPModel(
#             input_dim=512,
#             hidden_size=768,
#             num_hidden_layers=8,
#             intermediate_size=4 * 768)

#         preds = mlp(tokens)
#         error_cost = paddle.nn.functional.square_error_cost(preds, labels)
#         loss = paddle.mean(error_cost)

#     return train_program, start_program, loss


# def main():

#     from modeling_utils.ops import Topology
#     worker_num = paddle.distributed.get_world_size()
#     worker_index = paddle.distributed.get_rank()
    

#     dist_strategy = fleet.DistributedStrategy()
#     dist_strategy.amp = False
#     dist_strategy.pipeline = False
#     dist_strategy.recompute = False

#     # init parallel optimizer
#     dist_strategy.semi_auto = True
#     dist_strategy.auto_search = True
#     fleet.init(is_collective=True, strategy=dist_strategy)

#     if not dist_strategy.auto_search:
#         global _global_parallel_strategy
#         _global_parallel_stratergy = "dp"
#         global _global_process_mesh
#         _global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])

#     train_program = static.Program()
#     start_program = static.Program()
#     train_program, start_program, loss = mlp_forward(train_program, start_program)

#     # different from hybrid parallel
#     # optimizer = paddle.optimizer.Adam(
#     #     learning_rate=0.00001,
#     #     beta1=0.9,
#     #     beta2=0.999,
#     #     epsilon=1e-08,
#     #     grad_clip=None)
#     optimizer = paddle.optimizer.SGD(learning_rate=1e-4)

#     optimizer = fleet.distributed_optimizer(optimizer)
#     _, _, distributed_startup_program, distributed_main_program = optimizer.minimize(loss, start_program)
#     print(str(dist_strategy))

#     with open("./output" + "/auto_main_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
#         f.write(str(distributed_main_program))
#     with open("./output" + "/auto_startup_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
#         f.write(str(distributed_startup_program))

#     paddle.seed(worker_index + 2021)
#     random.seed(worker_index + 2021)
#     np.random.seed(worker_index + 2021)

#     place = paddle.set_device("gpu")
#     exe = paddle.static.Executor(place)
#     exe.run(distributed_startup_program)

#     inputs = np.random.random(size=(400, 512)).astype('float32')
#     labels = np.random.random(size=(400, 512)).astype('float32')

#     for step in range(150):
#         res = exe.run(distributed_main_program,
#                       feed={
#                           "tokens": inputs[step * 4:(step + 1) * 4, :],
#                           "labels": labels[step * 4:(step + 1) * 4, :]},
#                       fetch_list=[loss])
#         print("step: %d, loss_print: %f" % (step, res[0]))


# if __name__ == "__main__":
#     main()





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

import unittest
import random
import numpy as np
import os
import shutil

import paddle
import paddle.nn as nn
import paddle.utils as utils
import paddle.static as static
import paddle.nn.functional as F
import paddle.distributed.auto_parallel as auto

from paddle.distributed import fleet
from paddle.fluid.initializer import NumpyArrayInitializer
from paddle.distributed.auto_parallel.utils import save_distributed_checkpoint, load_distributed_checkpoint, load_checkpoint_into_program
from paddle.distributed.auto_parallel.utils import get_dist_attr, merge_and_slice_parameter, load_parameter_into_program


paddle.enable_static()
_global_parallel_strategy = None
_global_process_mesh = None
PP_MESH_0 = None
PP_MESH_1 = None


class MLPLayer(nn.Layer):
    def __init__(self,
                 hidden_size=64,
                 intermediate_size=4 * 64,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        np.random.seed(2021)
        arr0 = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
        arr1 = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
        weight_attr0 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr0))
        weight_attr1 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr1))
        bias_attr = None
        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr0, bias_attr=bias_attr)
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr1, bias_attr=bias_attr)
        self.linear2 = nn.Linear(
            d_model, dim_feedforward, weight_attr0, bias_attr=bias_attr)
        self.linear3 = nn.Linear(
            dim_feedforward, d_model, weight_attr1, bias_attr=bias_attr)
        self.linear4 = nn.Linear(
            d_model, dim_feedforward, weight_attr0, bias_attr=bias_attr)
        self.linear5 = nn.Linear(
            dim_feedforward, d_model, weight_attr1, bias_attr=bias_attr)
        self.norm0 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        # self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, input):
        out = self.norm0(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)

        out = self.norm1(out)
        out = self.linear2(out)
        out = F.gelu(out, approximate=True)
        out = self.linear3(out)

        # out = self.norm2(out)
        out = self.linear4(out)
        out = F.gelu(out, approximate=True)
        out = self.linear5(out)
        return out


def mlp_forward(train_program, start_program):
    with static.program_guard(train_program,start_program), \
        utils.unique_name.guard():
        batch_size = 4
        hidden_size = 64
        input = static.data(
            name="input", shape=[batch_size, hidden_size], dtype='float32')
        label = static.data(
            name="label", shape=[batch_size, 1], dtype='float32')
        input.stop_gradient=False

        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02)
        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)
    return loss, train_program, start_program


def get_distributed_program():
    train_program = static.Program()
    startup_program = static.Program()
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    dist_strategy.auto_search = True
    fleet.init(is_collective=True, strategy=dist_strategy)
    loss, train_program, startup_program = mlp_forward(train_program,
                                                       startup_program)

    with open("./output/autosearch2" + "/serial_main_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(train_program))
    with open("./output/autosearch2" + "/serial_startup_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(startup_program))

    optimizer = paddle.fluid.optimizer.SGDOptimizer(learning_rate=0.01)
    optimizer = fleet.distributed_optimizer(optimizer)
    _, _, dist_startup_prog, dist_main_prog = optimizer.minimize(
        loss, startup_program)

    return dist_main_prog, dist_startup_prog, loss


def main():
    paddle.seed(2021)
    random.seed(2021)
    np.random.seed(2021)

    input = np.random.random(size=(400, 64)).astype('float32')
    label = np.random.random(size=(400, 1)).astype('float32')

    dist_main_prog, dist_start_prog, loss = get_distributed_program()

    with open("./output/autosearch2" + "/auto_main_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(dist_main_prog))
    with open("./output/autosearch2" + "/auto_startup_program.txt.%d" % (paddle.distributed.get_rank()), 'w') as f:
        f.write(str(dist_start_prog))

    place = paddle.set_device("gpu")
    exe = paddle.static.Executor(place)
    exe.run(dist_start_prog)
    print("========================end of start up prog========================")

    ckpt_path = [
        "./output/mp4/step_10/model_state_rank0.pdmodel",
        "./output/mp4/step_10/model_state_rank1.pdmodel",
        "./output/mp4/step_10/model_state_rank2.pdmodel",
        "./output/mp4/step_10/model_state_rank3.pdmodel",
        ]
    dist_attr_path = [
        "./output/mp4/step_10/dist_attr_rank0.pdattr",
        "./output/mp4/step_10/dist_attr_rank1.pdattr",
        "./output/mp4/step_10/dist_attr_rank2.pdattr",
        "./output/mp4/step_10/dist_attr_rank3.pdattr",
    ]
    load_checkpoint_into_program(ckpt_path, dist_attr_path, dist_main_prog)
    print("========================end of load parameter========================")

    for step in range(10, 50):
        if step == 20:
            output_dir = "./output/autosearch2/" + "step_" + str(step) 
            os.makedirs(output_dir, exist_ok=True)
            save_distributed_checkpoint(
                dist_main_prog, output_dir, dist_attr_path=output_dir)

        res = exe.run(dist_main_prog,
                        feed={
                            "input": input[step * 4:(step + 1) * 4, :],
                            "label": label[step * 4:(step + 1) * 4, :]
                        },
                        fetch_list=[loss])
        print(step, " ", res[0])


if __name__ == "__main__":
    main()
