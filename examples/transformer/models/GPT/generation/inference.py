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
from ppfleetx.core.engine import InferenceEngine, TensorRTConfig
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

    if 'Inference' in config:
        inference_configs = config['Inference']
        inference_engine = None
    else:
        raise RuntimeError(f'No Inference in config')

    input_text = 'Hi, GPT2. Tell me who Jack Ma is.'
    input_ids = [tokenizer.encode(input_text)]

    if inference_engine is None:
        # parse TensorRT config
        tensorrt_config = None
        if 'TensorRT' in inference_configs:
            tensorrt_config = TensorRTConfig(**inference_configs['TensorRT'])

        inference_engine = InferenceEngine(inference_configs['model_dir'],
                                           inference_configs['mp_degree'],
                                           tensorrt_config)

    outs = inference_engine.predict([input_ids])

    ids = list(outs.values())[0]
    out_ids = [int(x) for x in ids[0]]
    result = tokenizer.decode(out_ids)
    result = input_text + result

    print('Prompt:', input_text)
    print('Generation:', result)
