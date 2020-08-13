# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


def parse_train_configs():
    parser = argparse.ArgumentParser("fleet-lightning")
    parser.add_argument(
        "--lr", type=float, default=0.1, help="base learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum value")
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta1 value for adam optimizer")
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.997,
        help="beta2 value for adam optimizer")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-9,
        help="epsilon value for adam optimizer")
    args = parser.parse_args()
    return args
