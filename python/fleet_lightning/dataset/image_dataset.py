# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import functools
import math
import numpy as np
import random
import paddle
import paddle.fluid as fluid
from .img_tool import process_image


def image_dataloader_from_filelist(filelist,
                                   inputs,
                                   batch_size=32,
                                   phase="train",
                                   shuffle=True,
                                   use_dali=False,
                                   use_mixup=False,
                                   image_mean=[0.485, 0.456, 0.406],
                                   image_std=[0.229, 0.224, 0.225],
                                   resize_short_size=256,
                                   lower_scale=0.08,
                                   lower_ratio=3. / 4,
                                   upper_ratio=4. / 3,
                                   data_layout='NHWC'):
    trainer_id = int(os.environ.get('PADDLE_TRAINER_ID'))
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    if not use_dali:
        loader = create_data_loader(inputs, phase, use_mixup)
        reader = reader_creator(
            filelist,
            phase,
            shuffle,
            image_mean,
            image_std,
            resize_short_size,
            lower_scale,
            lower_ratio,
            upper_ratio,
            data_layout=data_layout)
        batch_reader = paddle.batch(reader, batch_size)
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        places = fluid.CUDAPlace(gpu_id)
        loader.set_sample_list_generator(batch_reader, places)
    else:
        import dali
        loader = dali.train(
            filelist,
            batch_size,
            image_mean,
            image_std,
            resize_short_size,
            lower_scale,
            lower_ratio,
            upper_ratio,
            trainer_id=trainer_id,
            trainers_num=num_trainers,
            gpu_id=gpu_id,
            data_layout=data_layout)
    return loader


def reader_creator(filelist,
                   phase,
                   shuffle,
                   image_mean,
                   image_std,
                   resize_short_size,
                   lower_scale,
                   lower_ratio,
                   upper_ratio,
                   pass_id_as_seed=0,
                   data_layout='NHWC'):
    def _reader():
        data_dir = filelist[:-4]
        if not os.path.exists(data_dir):
            for i in range(len(filelist)):
                if filelist[-i] == '/':
                    file_root = filelist[:-i]
                    break
            data_dir = os.path.join(file_root, phase)
        with open(filelist) as flist:
            full_lines = [line.strip() for line in flist]
            if shuffle:
                if (not hasattr(_reader, 'seed')):
                    _reader.seed = pass_id_as_seed
                random.Random(_reader.seed).shuffle(full_lines)
                print("reader shuffle seed", _reader.seed)
                if _reader.seed is not None:
                    _reader.seed += 1

            if phase == 'train':
                trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
                if os.getenv("PADDLE_TRAINER_ENDPOINTS"):
                    trainer_count = len(
                        os.getenv("PADDLE_TRAINER_ENDPOINTS").split(","))
                else:
                    trainer_count = int(os.getenv("PADDLE_TRAINERS", "1"))

                per_node_lines = int(
                    math.ceil(len(full_lines) * 1.0 / trainer_count))
                total_lines = per_node_lines * trainer_count

                # aligned full_lines so that it can evenly divisible
                full_lines += full_lines[:(total_lines - len(full_lines))]
                assert len(full_lines) == total_lines

                # trainer get own sample
                lines = full_lines[trainer_id:total_lines:trainer_count]
                assert len(lines) == per_node_lines

                print("trainerid, trainer_count", trainer_id, trainer_count)
                print(
                    "read images from %d, length: %d, lines length: %d, total: %d"
                    % (trainer_id * per_node_lines, per_node_lines, len(lines),
                       len(full_lines)))
            else:
                print("mode is not train")
                lines = full_lines

            for line in lines:
                if phase == 'train':
                    img_path, label = line.split()
                    img_path = img_path.replace("JPEG", "jpeg")
                    img_path = os.path.join(data_dir, img_path)
                    yield (img_path, int(label))
                elif phase == 'val':
                    img_path, label = line.split()
                    img_path = img_path.replace("JPEG", "jpeg")
                    img_path = os.path.join(data_dir, img_path)
                    yield (img_path, int(label))

    image_mapper = functools.partial(
        process_image,
        mode=phase,
        color_jitter=False,
        rotate=False,
        crop_size=224,
        mean=image_mean,
        std=image_std,
        resize_short_size=resize_short_size,
        lower_scale=lower_scale,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        data_layout=data_layout)
    reader = paddle.reader.xmap_readers(
        image_mapper, _reader, 4, 4000, order=False)
    return reader


def create_data_loader(inputs, phase, use_mixup, data_layout='NHWC'):

    feed_image = inputs[0]

    feed_label = inputs[1]
    feed_y_a = fluid.data(
        name="feed_y_a", shape=[None, 1], dtype="int64", lod_level=0)

    if phase == 'train' and use_mixup:
        feed_y_b = fluid.data(
            name="feed_y_b", shape=[None, 1], dtype="int64", lod_level=0)
        feed_lam = fluid.data(
            name="feed_lam", shape=[None, 1], dtype="float32", lod_level=0)

        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[feed_image, feed_y_a, feed_y_b, feed_lam],
            capacity=64,
            use_double_buffer=True,
            iterable=True)
        return data_loader
    else:
        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[feed_image, feed_label],
            capacity=64,
            use_double_buffer=True,
            iterable=True)

        return data_loader
