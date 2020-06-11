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
        "--gpu_ids", type=str, default="0,1,2,3,4,5,6,7", help="training gpu")
    parser.add_argument(
        "--lr", type=float, default=0.00001, help="base learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.99, help="momentum value")
    args = parser.parse_args()
    return args
