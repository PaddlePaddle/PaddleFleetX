
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


export CUDA_VISIBLE_DEVICES=0

if [ $1 == "MNLI" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Data.Train.dataset.name=MNLI \
      -o Data.Train.dataset.root=./dataset/multinli_1.0 \
      -o Data.Eval.dataset.name=MNLI \
      -o Data.Eval.dataset.root=./dataset/multinli_1.0 \
      -o Data.Eval.dataset.split=dev_matched \
      -o Model.num_classes=3
elif [ $1 == "SST2" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Data.Train.dataset.name=SST2 \
      -o Data.Train.dataset.root=./dataset/SST-2/ \
      -o Data.Eval.dataset.name=SST2 \
      -o Data.Eval.dataset.root=./dataset/SST-2/ \
      -o Data.Eval.dataset.split=dev \
      -o Model.num_classes=2
elif [ $1 == "CoLA" ]
then
    python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
      -o Data.Train.dataset.name=CoLA \
      -o Data.Train.dataset.root=./dataset/cola_public/ \
      -o Data.Eval.dataset.name=CoLA \
      -o Data.Eval.dataset.root=./dataset/cola_public/ \
      -o Data.Eval.dataset.split=dev \
      -o Model.num_classes=2
else
   echo "Task name not recognized, please input MNLI, SST2, CoLA."
fi
