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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import numpy as np

from paddle.distributed import fleet
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from ppfleetx.utils import config, env
from ppfleetx.utils.log import logger
from ppfleetx.data import build_dataloader, tokenizers
from ppfleetx.models import build_module
from ppfleetx.core import EagerEngine

def save_images(images, output='', num_unets=1):
    """ save images"""
    if not os.path.exists(output):
        os.makedirs(output)
    img_size = [64, 512]
    for i in range(num_unets):
        for ith, image in enumerate(images[i]):
            norm_image = cv2.normalize(
                image.transpose([1, 2, 0]).cpu().numpy(),
                None, alpha = 0, beta = 255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            norm_image = norm_image.astype(np.uint8)[..., ::-1]
            img_gen = output + '/' + 'display.paper_' + str(
                img_size[i])+ '.' + str(ith) +'.jpg'
            cv2.imwrite(img_gen, norm_image)


if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)

    if dist.get_world_size() > 1:
        fleet.init(is_collective=True, strategy=env.init_dist_env(cfg))

    env.set_seed(cfg.Global.seed)
    module = build_module(cfg)
    config.print_config(cfg)

    tokenizer = tokenizers.get_t5_tokenizer(name='t5-11b')
    engine = EagerEngine(configs=cfg, module=module, mode='inference')

    input_text = 'one bike.'
    input_texts = [input_text] * 4
    encoded = tokenizer.batch_encode_plus(input_texts)
    input_ids = encoded.input_ids
    attn_masks = encoded.attention_mask
    input_dict = {'input_ids': input_ids, 'text_masks': attn_masks}

    img_outs = engine._module.forward(**input_dict)
    save_images(img_outs, 'test', num_unets=2)

