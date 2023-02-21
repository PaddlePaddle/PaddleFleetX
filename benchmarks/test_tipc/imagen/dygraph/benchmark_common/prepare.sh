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
wget -O projects/imagen/part-00079 https://paddlefleetx.bj.bcebos.com/data/laion400m/part-00079
# T5-11B
mkdir -p projects/imagen/t5/t5-11b/ && cd projects/imagen/t5/t5-11b/
wget https://paddlefleetx.bj.bcebos.com/tokenizers/t5/t5-11b/config.json
wget https://paddlefleetx.bj.bcebos.com/tokenizers/t5/t5-11b/spiece.model
wget https://paddlefleetx.bj.bcebos.com/tokenizers/t5/t5-11b/tokenizer.json
wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.0
wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.1
wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.2
wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.3
wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.4
cat t5.pd.tar.gz.* |tar -xf -
cd -
# DeBERTa V2 1.5B
mkdir -p projects/imagen/cache/deberta-v-xxlarge && cd projects/imagen/cache/deberta-v-xxlarge
wget https://paddlefleetx.bj.bcebos.com/tokenizers/debertav2/config.json
wget https://paddlefleetx.bj.bcebos.com/tokenizers/debertav2/spm.model
wget https://paddlefleetx.bj.bcebos.com/tokenizers/debertav2/tokenizer_config.json
wget https://fleetx.bj.bcebos.com/DebertaV2/debertav2.pd.tar.gz.0
wget https://fleetx.bj.bcebos.com/DebertaV2/debertav2.pd.tar.gz.1
cat debertav2.pd.tar.gz.* | tar -xf -
cd -
