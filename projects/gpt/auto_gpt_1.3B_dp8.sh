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

log_dir=log_auto_dp8sharding8
rm -rf $log_dir

# 1.3B+dp8 run_pretrain
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    ./tools/auto.py \
    -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp8.yaml \
    -o Model.use_recompute=True \
    -o Model.recompute_granularity= \
    -o Model.hidden_size=1024 \
    -o Model.num_layers=4 \
    -o Engine.max_steps=30 \
    -o Engine.logging_freq=1 \
    -o Engine.eval_freq=100 \
    -o Engine.save_load.output_dir="" \
    -o Engine.verbose=3 \
    -o Engine.mix_precision.level=o1 \
    -o Distributed.dp_degree=8 \
    -o Distributed.mp_degree=1 \
    -o Distributed.pp_degree=1 \
    -o Distributed.sharding.sharding_degree=8 \
    -o Distributed.sharding.sharding_stage=3 \


# export FLAGS_USE_STANDALONE_EXECUTOR=1
# export FLAGS_CONVERT_GRAPH_TO_PROGRAM=1
# export GLOG_v=0
# export FLAGS_new_executor_sequential_run=1
# export FLAGS_fraction_of_gpu_memory_to_use=0.1

# python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 \
#     tools/auto.py -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_6.7B_sharding16.yaml \
#     -o Global.global_batch_size=64 \
#     -o Global.local_batch_size=32 \
#     -o Global.micro_batch_size=8  \
#     -o Model.hidden_dropout_prob=0 \
#     -o Model.attention_probs_dropout_prob=0 \
#     -o Model.use_recompute=True \
#     -o Distributed.dp_degree=2 \
#     -o Distributed.mp_degree=4 \
#     -o Distributed.pp_degree=1 \
#     -o Distributed.sharding.sharding_degree=2 \
#     -o Distributed.sharding.sharding_stage=1/2/3 \
#     -o Engine.mix_precision.level=o1/o2/o3 \
#     -o Engine.max_steps=100 \
#     -o Engine.eval_freq=100000 \
#     -o Engine.verbose=3 \
#     -o Engine.logging_freq=1
