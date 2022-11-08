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

import paddle

import os
import sys
import numbers
import numpy as np

try:
    from collections.abc import Sequence, Mapping
except:
    from collections import Sequence, Mapping

from ppfleetx.data.sampler import Stack, Tuple


def collate_fn(batch):
    """
    Default batch collating function for :code:`paddle.io.DataLoader`,
    get input data as a list of sample datas, each element in list
    if the data of a sample, and sample data should composed of list,
    dictionary, string, number, numpy array and paddle.Tensor, this
    function will parse input data recursively and stack number,
    numpy array and paddle.Tensor datas as batch datas. e.g. for
    following input data:
    [{'image': np.array(shape=[3, 224, 224]), 'label': 1},
     {'image': np.array(shape=[3, 224, 224]), 'label': 3},
     {'image': np.array(shape=[3, 224, 224]), 'label': 4},
     {'image': np.array(shape=[3, 224, 224]), 'label': 5},]
    
    
    This default collate function zipped each number and numpy array
    field together and stack each field as the batch field as follows:
    {'image': np.array(shape=[4, 3, 224, 224]), 'label': np.array([1, 3, 4, 5])}
    Args:  
        batch(list of sample data): batch should be a list of sample data.
    
    Returns:
        Batched data: batched each number, numpy array and paddle.Tensor
                      in input data.
    """
    sample = batch[0]
    if isinstance(sample, np.ndarray):
        batch = np.stack(batch, axis=0)
        return batch
    elif isinstance(sample, paddle.Tensor):
        return paddle.stack(batch, axis=0)
    elif isinstance(sample, numbers.Number):
        batch = np.array(batch)
        return batch
    elif isinstance(sample, (str, bytes)):
        return batch
    elif isinstance(sample, Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in sample}
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            raise RuntimeError(
                "fileds number not same among samples in a batch")
        return [collate_fn(fields) for fields in zip(*batch)]

    raise TypeError("batch data con only contains: tensor, numpy.ndarray, "
                    "dict, list, number, but got {}".format(type(sample)))


def default_collate_fn(batch_transform=None):
    if batch_transform is not None:
        # batch_ops = create_preprocess_operators(batch_transform)

        # def inner_collate_fn(batch):
        #     batch = transform(batch, batch_ops)
        #     batch = collate_fn(batch)
        #     return batch

        # return inner_collate_fn
        pass
    else:
        return collate_fn


def gpt_collate_fn(batch):
    return Tuple([Stack() for raw in zip(*batch)])(batch)


class ErnieCollateData():
    def __init__(self, micro_batch_size=1):
        self.micro_batch_size = micro_batch_size

    def generate_data(self, data, stack_fn=Stack()):
        num_fields = len(data[0])
        out = [None] * num_fields
        # 0. input_ids,
        # 1. segment_ids,
        # 2. input_mask,
        # 3. masked_lm_positions,
        # 4. masked_lm_labels,
        # 5. next_sentence_labels
        for i in (0, 1, 2, 5):
            out[i] = stack_fn([x[i] for x in data])
        out[5] = out[5].reshape([-1, 1])
        batch_size, seq_length = out[0].shape
        size = num_mask = sum(len(x[3]) for x in data)
        # masked_lm_positions
        # Organize as a 1D tensor for gather or use gather_nd
        if size % 8 != 0:
            size += 8 - (size % 8)
        out[3] = np.full(size, 0, dtype=np.int32)

        # masked_lm_labels
        out[4] = np.full([size, 1], -1, dtype=np.int64)
        mask_token_num = 0
        for i, x in enumerate(data):
            for j, pos in enumerate(x[3]):
                out[3][mask_token_num] = i * seq_length + pos
                out[4][mask_token_num] = x[4][j]
                mask_token_num += 1
        return out

    def __call__(self, data):
        accumulate_steps = len(data) // self.micro_batch_size
        if accumulate_steps == 1:
            return self.generate_data(data)
        else:
            self.micro_batch_size = len(data) // accumulate_steps
            all_data = [[] for _ in range(6)]
            for acc_step in range(accumulate_steps):
                tmp = self.generate_data(
                    data[acc_step * self.micro_batch_size:(acc_step + 1) *
                         self.micro_batch_size])
                for i in range(6):
                    all_data[i].append(tmp[i])
            return all_data


def imagen_collate_fn(batch):
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
