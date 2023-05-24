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
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

import numpy as np
import paddle.distributed.fleet as fleet
from ppfleetx.data.tokenizers import GPTTokenizer
from ppfleetx.core.engine import InferenceEngine
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser("ernie inference")
    parser.add_argument(
        '-m', '--model_dir', type=str, default='./output', help='model dir')
    parser.add_argument(
        '-mp', '--mp_degree', type=int, default=1, help='mp degree')
    parser.add_argument(
        '-d', '--device', type=str, default='', help='device type')
    args = parser.parse_args()
    return args


def main(args):
    fleet.init(is_collective=True)
    infer_engine = InferenceEngine(
        args.model_dir, args.mp_degree, device=args.device)
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    text = 'Hi ERNIE. Tell me who Jack Ma is.'
    inputs = tokenizer(text, padding=True, return_attention_mask=True)

    whole_data = [
        np.array(inputs['token_type_ids']).reshape(1, -1),
        np.array(inputs['input_ids']).reshape(1, -1)
    ]

    start_time = time.time()
    tick_time = start_time
    for i in range(10000):
        outs = infer_engine.predict(whole_data)
        tok_time = time.time()
        loop_duration = tok_time - tick_time
        tick_time = tok_time
        avg_time = (tok_time - start_time) / (i + 1)
        if i % 10 == 0:
            print(
                "iter: {}/10000, time consumed: {:3f} (ms), avg time: {:3f} (ms)".
                format(i, loop_duration * 1000, avg_time * 1000))
    # outs = infer_engine.predict(whole_data)
    # print(outs)
    print("Done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
