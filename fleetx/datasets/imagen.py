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

import time
import os

import random
import base64
import numpy as np
import blobfile as bf

from random import randint, choice
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from copy import deepcopy
from PIL import Image, ImageFile

import paddle
from paddle.io import Dataset, DataLoader
from paddle.distributed import get_world_size
from paddle.vision import transforms as T


def collate_imagen(batch, tokenizer=None):
    """ collate for imagen base64 """
    text_embs = []
    images = []
    attn_masks = []
    max_len = 0
    for image, text_emb, attn_mask in batch:
        if text_emb is None:
            return [None] * 3
        text_len, dim = text_emb.shape
        if text_len > max_len:
            max_len = text_emb.shape[0]

        images.append(image)
        text_embs.append(text_emb)
        attn_masks.append(attn_mask)
    bsz = len(images)
    dim = text_embs[0].shape[-1]
    text_embeds = paddle.zeros(shape=[bsz, max_len, dim], dtype=np.float32)
    text_masks = paddle.zeros(shape=[bsz, max_len], dtype=np.int64)
    images = paddle.stack(images)
    for i, (emb, mask) in enumerate(zip(text_embs, attn_masks)):
        text_embeds[i, :emb.shape[0], :] = emb
        text_masks[i, :mask.shape[0]] = mask

    return images, text_embeds, text_masks


def get_files(data_path, gpu_num, shuffle=False):
    files = [
        file.strip() for file in open(data_path).readlines()
        if file.strip() != ""
    ]
    if shuffle:
        random.shuffle(files)
    local_rank = paddle.distributed.get_rank()

    if len(files) % gpu_num == 0:
        files_extend = list(files)
    else:
        added_num = gpu_num - (len(files) % gpu_num)
        try:
            files_extend = files + random.sample(files, added_num)
        except:
            files_extend = files + random.sample(files, 1) * added_num

    assert len(
        files_extend
    ) % gpu_num == 0  # make sure the files can be evenly distributed
    num_files_per_gpu = len(files_extend) / gpu_num

    files = files_extend[local_rank::gpu_num]
    print("files: ", files, local_rank)
    print("num files per gpu: ", num_files_per_gpu)

    return files


def create_imagen_dataloader(configs, places=None, worker_init=None):
    data_path = configs['input_path']
    shuffle = configs['shuffle']
    input_resolusion = configs['input_resolusion']
    text_max_len = configs['text_max_len']
    device_world_size = paddle.distributed.get_world_size()
    device_world_rank = paddle.distributed.get_rank()
    files = get_files(data_path, device_world_size, shuffle=shuffle)
    dataset_train = ImagenEmbedPairDataset(
        data_path=files,
        image_size=input_resolusion,
        text_max_len=text_max_len)
    return DataLoader(
        dataset_train,
        batch_size=configs['batch_size'],
        shuffle=shuffle,
        drop_last=True,
        num_workers=configs['num_workers'],
        collate_fn=collate_imagen,
        places=places,
        use_shared_memory=configs['use_shared_memory'],
        worker_init_fn=worker_init)


def data_augmentation_for_imagen(img, resolution):
    arr = deepcopy(img)
    while min(*arr.size) >= 2 * resolution:
        arr = arr.resize(
            tuple(x // 2 for x in arr.size), resample=Image.Resampling.BOX)
    scale = resolution / min(*arr.size)
    arr = arr.resize(
        tuple(round(x * scale) for x in arr.size),
        resample=Image.Resampling.BICUBIC)

    arr = np.array(arr.convert("RGB"))
    crop_y = (arr.shape[0] - resolution) // 2
    crop_x = (arr.shape[1] - resolution) // 2
    arr = arr[crop_y:crop_y + resolution, crop_x:crop_x + resolution]
    arr = arr.astype(np.float32)
    arr = np.transpose(arr, [2, 0, 1])
    return paddle.to_tensor(arr)


class ImagenEmbedPairDataset(Dataset):
    def __init__(self,
                 data_path,
                 input_format='embed_base64_cc12m',
                 image_size=64,
                 second_size=256,
                 text_max_len=128,
                 filter_image_resolution=128,
                 tokenizer=None,
                 split='train'):
        super().__init__()
        self.filter_image_resolution = filter_image_resolution
        self.image_size = image_size
        self.text_max_len = text_max_len
        self.split = split
        self.filename = data_path
        if not isinstance(self.filename, list):
            self.indexes = self.load_path(self.filename)
        else:
            self.indexes = []
            for f_index, f in enumerate(self.filename):
                self.load_path(f, f_index)
        self.legacy_index = []
        self.skip = 0
        self.tokenizer = tokenizer

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
    def load_file(filepath, filename):
        return np.load(os.path.join(filepath, filename), mmap_mode='r')

    @staticmethod
    def base64_to_image(base64_str):
        byte_data = base64.b64decode(base64_str)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)
        return img

    @staticmethod
    def get_line(filename, indexes):
        offset, n = indexes
        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            return f.readline()

    def __getitem__(self, index):
        if not isinstance(self.filename, list):
            data_dir = os.path.dirname(self.filename)
            data = self.get_line(self.filename, self.indexes[index])
        else:
            data_dir = os.path.dirname(self.filename[self.indexes[index][1]])
            data = self.get_line(self.filename[self.indexes[index][1]],
                                 self.indexes[index][0])
        data = data.strip().split('\t')
        text_embed = self.load_file(data_dir, data[1])
        attn_mask = self.load_file(data_dir, data[2])
        image = self.base64_to_image(data[3])
        image = data_augmentation_for_imagen(image, self.image_size)

        return image, paddle.to_tensor(
            text_embed, dtype='float32'), paddle.to_tensor(
                attn_mask, dtype='int64')

    def __len__(self):
        return len(self.indexes)
