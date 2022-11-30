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
import numpy as np
from PIL import Image
import paddle

from paddle.distributed import fleet
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from ppfleetx.utils import config
from ppfleetx.distributed.apis import env
from ppfleetx.utils.log import logger
from ppfleetx.data import build_dataloader, tokenizers
from ppfleetx.models import build_module
from ppfleetx.core import EagerEngine

def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)

def preprocess(img_path):
        """preprocess
        Preprocess to the input.
        Args: img_path: Image path.
        Returns: Input data after preprocess.
        """
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        # ResizeImage
        img = img.resize((224,224), Image.BILINEAR)

        # NormalizeImage
        scale = np.float32(1.0/255.0)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        shape = (1, 1, 3)
        mean = np.array(mean).reshape(shape).astype('float32')
        std = np.array(std).reshape(shape).astype('float32')
        img = (img * scale - mean) / std

        # ToNCHW
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)
    env.set_seed(cfg.Global.seed)
    np.random.seed(1)
    img_path = 'projects/vit/images/demo.jpg'
    img = preprocess(img_path)
    
    if(os.path.exists('shape.pbtxt')==False):
        cfg.Inference.TensorRT.collect_shape = True
        module = build_module(cfg)
        engine = EagerEngine(configs=cfg,module=module, mode='inference')
        outs = engine.inference([img])

    cfg.Inference.TensorRT.collect_shape = False
    module = build_module(cfg)
    config.print_config(cfg)
    engine = EagerEngine(configs=cfg,module=module, mode='inference')
    outs = engine.inference([img])
    res = softmax(outs['linear_99.tmp_1'])
    max_index = np.argmax(res, axis=-1)
    print("类型: ", max_index[0],)
    print("概率: ", res[0][max_index[0]])

    
