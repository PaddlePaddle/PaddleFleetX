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

import paddle

from ppfleetx.distributed.apis import io
from ppfleetx.utils.compression_helper import prune_model, quant_model


def compress_model(config, model, input_spec):
    quanter, quant_configs = None, None
    prune_configs, compress_configs = None, None

    if 'Compress' in config:
        compress_configs = config['Compress']
        if "Prune" in compress_configs:
            prune_configs = compress_configs["Prune"]
        if "Quantization" in compress_configs:
            quant_configs = compress_configs["Quantization"]

        # Load pretrained model before compression
        if 'pretrained' in compress_configs and compress_configs[
                'pretrained'] is not None:
            ckpt_dir = compress_configs['pretrained']
            io.load(
                ckpt_dir,
                model,
                optimizer=None,
                mode='quant',
                load_recovery=None)

            # Avoid loading again
            config.Global.save_load.ckpt_dir = None

        if prune_configs is not None and prune_configs.enable:
            prune_model(model, prune_configs, input_spec)

        # NOTE(minghaoBD): We haven't fully tested Prune+Quantization, so an "else if" is put here for separation.
        elif quant_configs is not None and quant_configs.enable:
            model, quanter = quant_model(model, quant_configs)

    return model, quanter
