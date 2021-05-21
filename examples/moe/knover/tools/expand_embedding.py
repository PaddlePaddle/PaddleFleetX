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
"""Expand an embedding's dimension."""

import argparse

import numpy as np
import paddle
import paddle.fluid as fluid


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--embedding_name", type=str, default="pos_embedding")
    parser.add_argument("--embedding_new_size", type=int, required=True)

    return parser.parse_args()


def main(args):
    prog_state = fluid.io.load_program_state(args.param_path)
    embedding_weight = prog_state[args.embedding_name]
    if args.embedding_new_size > embedding_weight.shape[0]:
        random_weight = np.random.normal(
            scale=0.02,
            size=(args.embedding_new_size - embedding_weight.shape[0],) + embedding_weight.shape[1:]
        )
        random_weight = np.clip(random_weight, -0.02, 0.02).astype(embedding_weight.dtype)
        print(f"convert old embedding {args.embedding_name}: {embedding_weight.shape} "
              f"-> {(args.embedding_new_size,) + embedding_weight.shape[1:]}")
        embedding_weight = np.concatenate([embedding_weight, random_weight])
    else:
        print(f"convert old embedding {args.embedding_name}: {embedding_weight.shape} "
              f"-> {(args.embedding_new_size,) + embedding_weight.shape[1:]}")
        embedding_weight = embedding_weight[:args.embedding_new_size]
    prog_state[args.embedding_name] = embedding_weight

    program = fluid.Program()
    for k in prog_state:
        weight = prog_state[k]
        param = program.global_block().create_parameter(
            shape=weight.shape,
            dtype=weight.dtype,
            name=k)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(program)

    for k in prog_state:
        param_tensor = fluid.global_scope().find_var(k).get_tensor()
        param_tensor.set(prog_state[k], exe.place)

    fluid.io.save_params(exe, args.save_path, main_program=program)
    return


if __name__ == "__main__":
    paddle.enable_static()
    args = setup_args()
    main(args)
