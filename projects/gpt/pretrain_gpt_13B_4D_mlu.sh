#! /bin/bash
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#export CNCL_ALL_CONNECTED_TOPO_MODE=0
export CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE=1
#export CNCL_MLULINK_DISABLE=1
#export GLOG_v=10
#export FLAGS_call_stack_level=2

# Usage:
# node1: bash pretrain_gpt_13B_4D_mlu.sh log_gpt_worker0 ip1,ip2
# node2: bash pretrain_gpt_13B_4D_mlu.sh log_gpt_worker1 ip1,ip2

LOG_DIR=${1:-"log_gpt_worker0"}
IPS=${2:-"10.9.115.54,10.9.29.94"}
DATASET=${3:-"./dataset/openweb"}
LOG_GFILE=log_gpt_13B

rm $LOG_DIR -rf
mkdir -p ${LOG_DIR}

python -m paddle.distributed.launch \
       --log_dir ${LOG_DIR} \
       --ips=${IPS} \
       --device 0,1,2,3,4,5,6,7 tools/train.py \
       -c ppfleetx/configs/nlp/gpt/pretrain_gpt_13B_dp8.yaml \
       -o Global.device=mlu \
       -o Engine.mix_precision.level="O1" \
       -o Data.Train.dataset.input_dir=${DATASET} \
       -o Data.Eval.dataset.input_dir=${DATASET} \
       -o Data.Train.dataset.max_seq_len=1024 \
       -o Data.Eval.dataset.max_seq_len=1024 \
       -o Model.use_recompute=Fasle > ${LOG_DIR}/${LOG_GFILE} 2>&1 &
