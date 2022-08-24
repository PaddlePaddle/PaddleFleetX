# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import math
import os
import random
import time
import sys
import yaml
import numpy as np

import paddle
from examples.gpt.gpt_module import GPTGenerationModule
from examples.gpt.tools import parse_args, parse_yaml


def jit_export_GPT(module, model_path, configs): 
    from paddle.jit import save, load
    args = configs
    save(module.model, model_path, input_spec=[
        paddle.static.InputSpec(name='input_ids', shape=[-1, -1], dtype="int64"), 
        #args['max_dec_len'],
        1,
        args['min_dec_len'],
        args['decode_strategy'],
        args['temperature'],
        args['top_k'],
        args['top_p'],
        1.0,
        args['num_beams'],
        1,
        args['length_penalty'],
        args['early_stopping'],
        module.tokenizer.eos_token_id,
        module.tokenizer.eos_token_id,
        module.tokenizer.eos_token_id,
        None,
        None,
        None,
        args['num_return_sequences'],
        0.0,
        True,]
    )
    return None

def do_generation(to_static=True):
    configs = parse_yaml(parse_args())

    paddle.set_device(configs['Global']['device'])

    seed = configs['Global']['seed']

    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


    module = GPTGenerationModule(configs)
    model_dict = paddle.load("weights")
    module.model.set_state_dict(model_dict)

    #paddle.save(module.model.state_dict(), "weights")

    if to_static: 
        jit_export_GPT(module, "model_", module.configs)
        module.model = paddle.jit.load("model_")
        #module.model = paddle.jit.to_static(module.model)

    input_text = 'Where are you from?'
    result = module.generate(input_text)

    print("OutputIs", result[0])


if __name__ == "__main__":
    do_generation(False)
    do_generation(True)
