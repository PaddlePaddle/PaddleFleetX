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

tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}

OUTPUT_DIR='./outputs/'$my_name
echo $OUTPUT_DIR

DATA_PATH='data/cc12m_base64.lst'
PYTHON=python3.7

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# PADDLE perf flags
export FLAGS_gemm_use_half_precision_compute_type=False
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_use_autotune=True
export FLAGS_conv_workspace_size_limit=-1


# ============================ super resolution 64x64->256x256 ============================
$PYTHON -m paddle.distributed.launch \
  --gpus="0,1,2,3,4,5,6,7" \
  tools/run_imagen_text2im.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --log_dir ${OUTPUT_DIR} \
  --input_format embed_base64_cc \
  --shuffle \
  --model imagen_SR256 \
  --unet_num 1 \
  --channels 3 \
  --batch_size 8 --lr 1e-4 --warmup_epochs 2 --epochs 68 \
  --input_resolution 256 \
  --timesteps 1000 \
  --condition_on_text \
  --cond_drop_prob 0.1\
  --lowres_noise_schedule linear \
  --lowres_sample_noise_level 0.2 \
  --continuous_times \
  --auto_normalize_img \
  --dynamic_thresholding \
  --dynamic_thresholding_percentile 0.9 \
  --p2_loss_weight_gamma 0.5 \
  --p2_loss_weight_k 1.0 \
  --noise_schedules cosine \
  --pred_objectives noise \
  --loss_type l2 \
  --num_workers 8 \
  --seed 2022 \
  --distributed \
  --sharding_stage 2 \
  --sharding_degree 8 \
  --exp_name $my_name 
