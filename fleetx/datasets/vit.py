# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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
import copy
import numpy as np
import paddle
from fleetx.utils import logger
from fleetx.data.transforms.transform_utils import create_preprocess_operators, transform


def build_dataloader(config, mode, device):
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Dataset mode should be Train, Eval, Test"

    # build dataset
    class_num = config.get("class_num", None)
    config_dataset = config[mode]['dataset']
    config_dataset = copy.deepcopy(config_dataset)
    dataset_name = config_dataset.pop('name')
    batch_transform = config_dataset.pop('batch_transform_ops', None)
    dataset = ImageNetDataset(**config_dataset)

    # build sampler
    config_sampler = config[mode]['sampler']
    config_sampler = copy.deepcopy(config_sampler)
    sampler_name = config_sampler.pop("name")
    batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                      **config_sampler)

    # build dataloader
    config_loader = config[mode]['loader']
    num_workers = config_loader["num_workers"]
    use_shared_memory = config_loader["use_shared_memory"]

    data_loader = paddle.io.DataLoader(
        dataset=dataset,
        places=device,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=use_shared_memory,
        batch_sampler=batch_sampler,
        collate_fn=None)

    return data_loader


class ImageNetDataset(paddle.io.Dataset):
    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 delimiter=" ",
                 multi_label=False,
                 class_num=None):
        if multi_label:
            assert class_num is not None, "Must set class_num when multi_label=True"
        self.multi_label = multi_label
        self.classes_num = class_num

        self._img_root = image_root
        self._cls_path = cls_label_path
        self.delimiter = delimiter
        if transform_ops:
            self._transform_ops = create_preprocess_operators(transform_ops)

        self.images = []
        self.labels = []
        self._load_anno()

    def _load_anno(self, seed=None):
        assert os.path.exists(
            self._cls_path), f"{self._cls_path} does not exists"
        assert os.path.exists(
            self._img_root), f"{self._img_root} does not exists"
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for l in lines:
                l = l.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, l[0]))
                if self.multi_label:
                    self.labels.append(l[1])
                else:
                    self.labels.append(np.int32(l[1]))
                assert os.path.exists(self.images[-1])

    def __getitem__(self, idx):
        try:
            with open(self.images[idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            if self.multi_label:
                one_hot = np.zeros([self.classes_num], dtype=np.float32)
                cls_idx = [int(e) for e in self.labels[idx].split(',')]
                for idx in cls_idx:
                    one_hot[idx] = 1.0
                return (img, onehot)
            else:
                return (img, np.int32(self.labels[idx]))

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        if self.multi_label:
            return self.classes_num
        return len(set(self.labels))
