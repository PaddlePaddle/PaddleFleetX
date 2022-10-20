#!/usr/bin/env bash

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

# imagen text to image 512x512
# export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_gemm_use_half_precision_compute_type=True
export CUDA_VISIBLE_DEVICES=1
GLOG_v=0 python3.8 tasks/imagen/inference.py -c ./ppfleetx/configs/multimodal/imagen/imagen_text2im_64x64_to_512x512.yaml --infer --trt --half
