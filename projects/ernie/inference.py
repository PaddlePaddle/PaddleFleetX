# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from paddle.distributed import fleet
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

import numpy as np
import paddle.distributed.fleet as fleet
from ppfleetx.data.tokenizers import GPTTokenizer
from ppfleetx.core.engine import InferenceEngine
import argparse


def parse_args():
    parser = argparse.ArgumentParser("ernie inference")
    parser.add_argument(
        '-m', '--model_dir', type=str, default='./output', help='model dir')
    parser.add_argument(
        '-mp', '--mp_degree', type=int, default=1, help='mp degree')
    args = parser.parse_args()
    return args


def main(args):
    fleet.init(is_collective=True)
    infer_engine = InferenceEngine(args.model_dir, args.mp_degree)

    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    text = 'Hi ERNIE. Tell me who Jack Ma is.'
    inputs = tokenizer(text, padding=True, return_attention_mask=True)

    whole_data = [
        np.array(inputs['token_type_ids']).reshape(1, -1),
        np.array(inputs['input_ids']).reshape(1, -1)
    ]
    outs = infer_engine.predict(whole_data)
    print(outs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
