#!/usr/bin/python
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

# There are 13 integer features and 26 categorical features
import os
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet

continous_features = range(1, 14)
categorial_features = range(14, 40)
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


def get_dataloader(inputs,
                   train_files_path,
                   sparse_feature_dim,
                   batch_size,
                   shuffle=True):
    file_list = [
        str(train_files_path) + "/%s" % x for x in os.listdir(train_files_path)
    ]

    loader = fluid.io.DataLoader.from_generator(
        feed_list=inputs, capacity=64, use_double_buffer=True, iterable=True)
    train_generator = CriteoDataset(sparse_feature_dim)
    reader = train_generator.train(file_list,
                                   fleet.worker_num(), fleet.worker_index())
    if shuffle:
        reader = paddle.batch(
            paddle.reader.shuffle(
                reader, buf_size=batch_size * 100),
            batch_size=batch_size)
    else:
        reader = paddle.batch(reader, batch_size=batch_size)
    places = fluid.CPUPlace()
    loader.set_sample_list_generator(reader, places)
    return loader


class CriteoDataset(object):
    def __init__(self, sparse_feature_dim):
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [
            20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50
        ]
        self.cont_diff_ = [
            20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50
        ]
        self.hash_dim_ = sparse_feature_dim
        # here, training data are lines with line_index < train_idx_
        self.train_idx_ = 41256555
        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)

    def _reader_creator(self, file_list, is_train, trainer_num, trainer_id):
        def reader():
            for file in file_list:
                with open(file, 'r') as f:
                    line_idx = 0
                    for line in f:
                        line_idx += 1
                        features = line.rstrip('\n').split('\t')
                        dense_feature = []
                        sparse_feature = []
                        for idx in self.continuous_range_:
                            if features[idx] == '':
                                dense_feature.append(0.0)
                            else:
                                dense_feature.append(
                                    (float(features[idx]) -
                                     self.cont_min_[idx - 1]) /
                                    self.cont_diff_[idx - 1])
                        for idx in self.categorical_range_:
                            sparse_feature.append([
                                hash(str(idx) + features[idx]) % self.hash_dim_
                            ])

                        label = [int(features[0])]
                        yield [dense_feature] + sparse_feature + [label]

        return reader

    def train(self, file_list, trainer_num, trainer_id):
        return self._reader_creator(file_list, True, trainer_num, trainer_id)

    def test(self, file_list):
        return self._reader_creator(file_list, False, 1, 0)
