#! /bin/bash
# Runs the "1.3B" parameter model
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

log_dir=log_hybrid
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/qat_gpt_6.7B_sharding16.yaml \
    -o Engine.max_steps=100000 \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Optimizer.lr.decay_steps=72000 \
    -o Optimizer.weight_decay=0.02 \
    -o Optimizer.lr.max_lr=5.0e-6 \
    -o Optimizer.lr.min_lr=1.0e-6 \
    -o Compress.pretrained='./PaddleFleetX_GPT_6.7B'
