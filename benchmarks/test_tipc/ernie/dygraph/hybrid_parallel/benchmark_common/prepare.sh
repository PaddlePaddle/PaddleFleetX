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

python -m pip install -r ../requirements.txt
# get data
cd ../
rm -rf dataset/ernie
mkdir -p dataset/ernie
wget -O dataset/ernie/cluecorpussmall_14g_1207_ids_part0 https://paddlefleetx.bj.bcebos.com/model/nlp/ernie/cluecorpussmall_14g_1207_ids_part0
wget -O dataset/ernie/cluecorpussmall_14g_1207_ids_part1 https://paddlefleetx.bj.bcebos.com/model/nlp/ernie/cluecorpussmall_14g_1207_ids_part1
cat dataset/ernie/cluecorpussmall_14g_1207_ids_part* &> dataset/ernie/cluecorpussmall_14g_1207_ids.npy
wget dataset/ernie/cluecorpussmall_14g_1207_idx.npz https://paddlefleetx.bj.bcebos.com/model/nlp/ernie/cluecorpussmall_14g_1207_idx.npz
