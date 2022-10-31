
#! /bin/bash

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
export PYTHONPATH=$PYTHONPATH:/workspace/workspace/PaddleNLP/

# export CUDA_VISIBLE_DEVICES=1
# python tools/train.py -c ppfleetx/configs/nlp/ernie/pretrain_ernie_base_single_card.yaml 

log_dir=log_hybrid
rm -rf $log_dir

# 1.3B+dp8 run_pretrain
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    ./tools/train.py \
    -c ppfleetx/configs/nlp/ernie/pretrain_ernie_base_3D.yaml
