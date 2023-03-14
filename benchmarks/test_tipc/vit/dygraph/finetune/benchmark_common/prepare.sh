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

python -m pip install -r ../requirements.txt
# get data
cd ../
mkdir dataset && cd dataset
cp -r ${BENCHMARK_ROOT}/models_data_cfs/Paddle_distributed/ILSVRC2012.tgz ./
tar -zxf ILSVRC2012.tgz
cd -

# pretrained
mkdir -p pretrained/vit/
wget -O ./pretrained/vit/imagenet21k-ViT-L_16.pdparams \
https://paddle-wheel.bj.bcebos.com/benchmark/imagenet21k-ViT-L_16.pdparams
