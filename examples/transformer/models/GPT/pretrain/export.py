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
from paddle.static import InputSpec

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../../../')))

from ppfleetx.distributed.apis import env, strategy, io
from ppfleetx.utils.log import logger
from ppfleetx.utils import device, log
from ppfleetx.utils.export import export_inference_model
from examples.transformer.utils import qat
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
    cfg.print_config(config)

    if config.Global.mix_precision.use_pure_fp16:
        logger.info("NOTE: disable use_pure_fp16 in export mode")

    # build GPT model
    model, _, _ = impls.build_model(config)

    # export
    model.eval()
    input_spec = [
        InputSpec(
            shape=[None, None], name="tokens", dtype='int64'), InputSpec(
                shape=[None, None], name="ids", dtype='int64')
    ]

    output_dir = config.Global.save_load.output_dir
    dp_rank = 0 if nranks == 1 else env.get_hcg().get_data_parallel_rank()
    save_dir = os.path.join(output_dir, "rank_{}".format(dp_rank))

    quanter = None
    quant_mode = False

    if 'Compress' in config:
        mode = 'compress'
        compress_configs = config['Compress']

        if "Quantization" in compress_configs:
            quant_mode = True

        model, quanter = qat.compress_model(config, model, input_spec)

    # load pretrained checkpoints
    if config.Global.save_load.ckpt_dir is not None:
        io.load(
            config.Global.save_load.ckpt_dir,
            model,
            optimizer=None,
            mode='export',
            load_recovery=None)

    if not quant_mode:
        export_inference_model(model, input_spec, save_dir, 'model')
    else:
        logger.info("export quantized model.")
        export_inference_model(
            model,
            input_spec,
            save_dir,
            'model',
            export_quant_model=True,
            quanter=quanter)
