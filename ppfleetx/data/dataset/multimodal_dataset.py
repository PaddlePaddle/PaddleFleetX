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

import os
import time
import gzip

import random
import base64
import numpy as np
import blobfile as bf

from random import randint, choice
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from copy import deepcopy
import PIL
from PIL import Image, ImageFile

import paddle
from paddle.io import Dataset, DataLoader
from paddle.distributed import get_world_size
from paddle.vision import transforms as T

from ppfleetx.utils.log import logger


def get_keys(data_path, gpu_num):
    files = [
        file.strip() for file in open(data_path).readlines()
        if file.strip() != ""
    ]
    local_rank = paddle.distributed.get_rank()

    if len(files) % gpu_num == 0:
        keys_extend = list(files)
    else:
        added_num = gpu_num - (len(files) % gpu_num)
        try:
            keys_extend = files + random.sample(files, added_num)
        except:
            keys_extend = files + random.sample(files, 1) * added_num

    keys = keys_extend[local_rank::gpu_num]
    logger.info("keys: {} {}".format(keys, local_rank))

    return keys


class ImagenDataset(Dataset):
    def __init__(self,
                 input_path,
                 image_format='base64',
                 shuffle=False,
                 image_size=64,
                 text_max_len=128,
                 filter_image_resolution=128,
                 tokenizer=None,
                 sr=False,
                 split='train',
                 interpolation="bicubic",
                 flip_p=0.5):
        super().__init__()
        device_world_size = paddle.distributed.get_world_size()
        self.filename = get_keys(input_path, gpu_num=device_world_size)
        if shuffle:
            random.shuffle(self.filename)
        self.filter_image_resolution = filter_image_resolution
        self.text_max_len = text_max_len
        self.split = split
        self.tokenizer = tokenizer
        self.sr = sr
        if sr:
            self.transform = T.Compose([T.Resize(image_size), T.ToTensor()])

        self.for_line = self.get_line_for_line(self.filename).__iter__()

        self.good_index = []

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip = T.RandomHorizontalFlip(prob=flip_p)
        self.image_size = image_size

    def load_path(self, data_path, f_index=None):
        if f_index is None:
            offset = 0
            with open(data_path, 'rb') as f:
                for line in tqdm(f, desc='Loading data'):
                    self.indexes.append((offset, len(line)))
                    offset += len(line)
        else:
            offset = 0
            with open(data_path, 'rb') as f:
                for line in tqdm(f, desc='Loading data'):
                    self.indexes.append(((offset, len(line)), f_index))
                    offset += len(line)

        if self.split == 'train':
            random.shuffle(self.indexes)
        return

    @staticmethod
    def base64_to_image(base64_str):
        byte_data = base64.b64decode(base64_str)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def get_line_for_line(self, filename):
        while True:
            for fname in filename:
                if fname[-2:] != "gz":
                    file = open(fname)
                    for line in file:
                        if line != "":
                            data = line.strip().split('\t')
                            image_base64 = data[4]
                            image_item = self.base64_to_image(image_base64)
                            if min(image_item.size) >= self.image_size:
                                yield line
                else:
                    file = gzip.GzipFile(fname, "r")
                    for line in file:
                        if line != "":
                            line = line.decode()
                            data = line.strip().split('\t')
                            image_base64 = data[4]
                            image_item = self.base64_to_image(image_base64)
                            if min(image_item.size) >= self.image_size:
                                yield line

    def __getitem__(self, index):
        if not isinstance(self.filename, list):
            data = self.for_line.__next__()
        else:
            data = self.for_line.__next__()

        data = data.strip().split('\t')

        # For laion 400m
        if len(data) == 6:
            image_base64 = data[4]
            caption = data[2]

        image_item = self.base64_to_image(image_base64)

        # Filter image resolution
        if min(image_item.size) < self.filter_image_resolution:
            return None

        if not self.sr:
            self.transform = T.Compose([
                T.CenterCrop([min(image_item.size), min(image_item.size)]),
                T.Resize(64), T.ToTensor()
            ])
            image_item = self.transform(image_item)
        else:
            img = np.array(image_item).astype(np.uint8)

            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]

            if img.shape[0] > img.shape[1]:
                img = img[0:crop, (w - crop) // 2:(w + crop) // 2]
            else:
                img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(
                    w + crop) // 2]

            image = Image.fromarray(img)
            image = image.resize(
                (self.image_size, self.image_size),
                resample=self.interpolation)

            image_item = self.transform(image)

        example = {'id': index, 'image': image_item, 'caption': caption}
        return example

    def __len__(self):
        #return len(self.indexes)
        if self.sr:
            return 300000000
        return 5000000
