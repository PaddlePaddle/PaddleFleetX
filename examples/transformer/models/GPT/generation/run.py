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

import os
import sys
import copy

import paddle
from paddle.distributed import fleet
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../../../')))

from ppfleetx.distributed.apis import env, strategy, io
from ppfleetx.utils.log import logger
from ppfleetx.utils import device, log
from examples.transformer.utils import config as cfg
from examples.transformer.utils import components as cpn

import impls

if __name__ == "__main__":
    # parse config from yaml
    args = cfg.parse_args()
    config = cfg.get_config(args.config, overrides=args.override, show=False)

    paddle.set_device(config.Global.device)

    # init distributed env
    nranks = dist.get_world_size()
    if nranks > 1:
        env.init_dist_env(config)

    env.set_seed(config.Global.seed)
    cfg.process_configs(config)

    # build model
    model, tokenizer = impls.build_model(config)
    model.eval()
    cfg.print_config(config)

    # call fleet wrapper
    if nranks > 1:
        model, _, _ = strategy.wrap_with_fleet(
            config.Distributed, model, optimizer=None, scaler=None)

    # load pretrained checkpoints
    if config.Global.save_load.ckpt_dir is not None:
        io.load(
            config.Global.save_load.ckpt_dir,
            model,
            optimizer=None,
            mode='generation',
            load_recovery=None)

    # build profiler
    if config.get('Profiler', {}).get('enable', False):
        profiler = cpn.build_profiler(config.Profiler)
    else:
        profiler = None

    input_text = 'Hi, GPT2. Tell me who Jack Ma is.'
    input_ids = tokenizer.encode(input_text)
    inputs = {'input_ids': [input_ids]}

    inputs = impls.left_padding(inputs, tokenizer.eos_token_id)
    input_ids = inputs['input_ids']

    if len(input_ids) == 0:
        input_ids = None
    else:
        # [1, seq_len]
        input_ids = paddle.to_tensor(input_ids, dtype='int64')

    ids, scores = model(input_ids=input_ids)

    result = []
    for i, generated_ids in enumerate(ids):
        generated_ids = generated_ids.numpy().tolist()
        # Decode text
        text = tokenizer.convert_ids_to_string(generated_ids)
        sequence = input_text + text
        result.append(sequence)

    print(f'Prompt: {input_text}')
    print(f'Generation: {result[0]}')

    if profiler:
        cpn.profiler_done(profiler, config.Profiler)
