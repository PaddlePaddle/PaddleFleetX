# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division

import os
import math
import pickle
import numpy as np
import random

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.paddle import DALIGenericIterator

import paddle
from paddle import fluid


IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

def convert_data_layout(data_layout):
    if data_layout == 'NCHW':
        return types.NCHW
    elif data_layout == 'NHWC':
        return types.NHWC
    else:
        raise ValueError("Not supported data_layout: {}".format(data_layout))


class HybridTrainPipe(Pipeline):
    """
    Create training pipeline.
    For more information please refer:
    https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/plugins/paddle_tutorials.html
    Note: You may need to find the newest DALI version.
    """
    def __init__(self,
                 file_root,
                 file_list,
                 batch_size,
                 crop,
                 min_area,
                 lower,
                 upper,
                 interp,
                 mean,
                 std,
                 device_id,
                 shard_id=0,
                 num_shards=1,
                 random_shuffle=True,
                 num_threads=4,
                 seed=42,
                 eii=None,
                 data_layout="NCHW"):
        super(HybridTrainPipe, self).__init__(batch_size,
                                              num_threads,
                                              device_id,
                                              seed=seed,
                                              prefetch_queue_depth=8)
        self.input = ops.FileReader(file_root=file_root,
                                    file_list=file_list,
                                    shard_id=shard_id,
                                    num_shards=num_shards,
                                    random_shuffle=random_shuffle)
        device_memory_padding = 211025920
        host_memory_padding = 140544512
        self.decode = ops.ImageDecoderRandomCrop(
                                    device='mixed',
                                    output_type=types.RGB,
                                    device_memory_padding=device_memory_padding,
                                    host_memory_padding=host_memory_padding,
                                    random_aspect_ratio=[lower, upper],
                                    random_area=[min_area, 1.0],
                                    num_attempts=10)
        self.res = ops.Resize(device='gpu',
                              resize_x=crop,
                              resize_y=crop,
                              interp_type=interp)
        output_layout = convert_data_layout(data_layout)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=output_layout,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=mean,
                                            std=std)
        self.coin = ops.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.to_int64(labels.gpu())]

    def __len__(self):
        return self.epoch_size("Reader")


def build(data_dir,
          batch_size,
          mode='train',
          trainer_id=None,
          trainers_num=None,
          lower_scale=0.08,
          gpu_id=0,
          epoch_id=0,
          image_shape="3,224,224",
          data_layout='NCHW'):
    env = os.environ
    assert float(env.get('FLAGS_fraction_of_gpu_memory_to_use', 0.92)) < 0.9, \
        "Please leave enough GPU memory for DALI workspace, e.g., by setting" \
        " `export FLAGS_fraction_of_gpu_memory_to_use=0.8`"
    file_root = data_dir
    batch_size = batch_size
    print("batch_size:", batch_size)

    mean = [v * 255.0 for v in IMAGE_MEAN]
    std = [v * 255.0 for v in IMAGE_STD]
    image_shape = [int(m) for m in image_shape.split(",")]
    crop = image_shape[1]
    min_area = lower_scale
    lower = 3.0 / 4
    upper = 4.0 / 3

    interp = types.INTERP_LANCZOS3

    file_list = os.path.join(file_root, 'train_list.txt')
    if not os.path.exists(file_list):
        raise ValueError("train_list.txt does not exist in {}.".format(
                          file_root))

    assert trainer_id is not None and trainers_num is not None, \
            "Please set trainer_id and trainers_num."
    print("dali gpu_id: {}, shard_id: {}, num_shards: {}".format(
                                                                 gpu_id,
                                                                 trainer_id,
                                                                 trainers_num))
    shard_id=trainer_id
    num_shards=trainers_num
    pipe = HybridTrainPipe(file_root,
                           file_list,
                           batch_size,
                           crop,
                           min_area,
                           lower,
                           upper,
                           interp,
                           mean,
                           std,
                           device_id=gpu_id,
                           shard_id=shard_id,
                           num_shards=num_shards,
                           seed=epoch_id,
                           data_layout=data_layout,
                           num_threads=4)
    pipe.build()
    pipelines = [pipe]
    sample_per_shard = len(pipe) // num_shards
    
    return DALIGenericIterator(pipelines,
                               ['train_image', 'feed_label'],
                               size=sample_per_shard)


def train(data_dir,
          batch_size,
          trainer_id=None,
          trainers_num=None,
          gpu_id=0,
          lower_scale=0.08,
          epoch_id=0,
          image_shape="3,224,224",
          data_layout="NCHW"):
    return build(data_dir,
                 batch_size,
                 'train',
                 trainer_id=trainer_id,
                 trainers_num=trainers_num,
                 gpu_id=gpu_id,
                 epoch_id=epoch_id,
                 image_shape=image_shape,
                 lower_scale=lower_scale,
                 data_layout=data_layout)

