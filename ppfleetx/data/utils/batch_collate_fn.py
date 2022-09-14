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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

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
    return Tuple(Stack(), Stack(), Stack(), Stack())(batch)


def gpt_eval_collate_fn(batch):
    return Tuple(Stack(), Stack(), Stack(), Stack(), Stack(), Stack())(batch)


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
