#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Read tf_records for pre-training."""
from __future__ import print_function
from __future__ import division

import os
import sys
import random
import numpy as np
import multiprocessing
import paddle as paddle
sys.path.append("..")
from tokenization import load_vocab

class DataReader(object):
    def __init__(self,
                 data_dir,
                 batch_size=4096,
                 in_tokens=True,
                 max_seq_len=128,
                 max_preds_per_seq=20,
                 shuffle_files=True,
                 is_test=False,
                 epoch=100,
                 **kwargs):
        self.data_dir = data_dir
        self.file_list = [data_dir + "/" + x for x in os.listdir(data_dir)]
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.epoch = epoch
        self.shuffle = shuffle_files
        self.in_tokens = in_tokens
        self.is_test = is_test

    def convert_samples_to_fluid_tensors(self, samples):
        src_ids = []
        sent_ids = []
        labels = []
        input_mask = []
        masked_lm_weight = []
        masked_lm_ids = []
        masked_lm_pos = []
        batch_size = len(samples)
        pos_ids = np.linspace(0, self.max_seq_len - 1, self.max_seq_len)
        pos_ids = np.tile(pos_ids, (batch_size, 1))
        
        def to_int_array(array_str):
            array_list = array_str.split()
            return [int(x) for x in array_list]

        def to_float_array(array_str):
            array_list = array_str.split()
            return [float(x) for x in array_list]

        for sample in samples:
            sample = sample.strip()
            group = sample.split(";")
            src_ids.extend(to_int_array(group[0]))
            sent_ids.extend(to_int_array(group[1]))
            input_mask.extend(to_int_array(group[2]))
            masked_lm_pos.extend(to_int_array(group[3]))
            masked_lm_ids.extend(to_int_array(group[4]))
            masked_lm_weight.extend(to_float_array(group[5]))
            labels.extend(to_int_array(group[6]))
            
        src_ids = np.array(src_ids).reshape(batch_size,
                                            self.max_seq_len,
                                            1)
        sent_ids = np.array(sent_ids).reshape(batch_size,
                                              self.max_seq_len,
                                              1)
        input_mask = \
                np.array(input_mask).astype("int64").reshape(
                    batch_size, self.max_seq_len)
        labels = np.array(labels).astype("int64").reshape(batch_size, 1)
        pos_ids = input_mask * pos_ids
        pos_ids = np.expand_dims(pos_ids, axis=-1).astype("int64")
        input_mask = np.expand_dims(input_mask, axis=-1).astype("float32")
        masked_lm_weight = \
                np.expand_dims(np.array(masked_lm_weight), axis=-1).astype("float32")
        masked_lm_ids = \
                np.expand_dims(np.array(masked_lm_ids), axis=-1).astype("int64")
        masked_lm_pos = \
                np.expand_dims(np.array(masked_lm_pos), axis=-1).astype("int64")

        masked_cnt = masked_lm_weight.sum(axis=1).astype("int").flatten()
        
        mask_label = np.empty((1, 1))
        mask_pos = np.empty((1, 1))
        for i, cnt in enumerate(masked_cnt):
            mask_label = np.append(mask_label, masked_lm_ids[i, 0:cnt])
            mask_pos = np.append(mask_pos,
                                 masked_lm_pos[i, 0:cnt] + i * self.max_seq_len)

        mask_label = mask_label.reshape([-1, 1]).astype("int64")
        mask_pos = mask_pos.reshape([-1, 1]).astype("int64")

        return src_ids, pos_ids, sent_ids, input_mask, mask_label, mask_pos, labels

    def load_samples(self, filename, shuffle):
        samples = []
        with open(filename) as fin:
            for line in fin:
                samples.append(line)
        if shuffle:
            random.shuffle(samples)
        return samples

    def get_progress(self):
        return self.total_file, self.current_file_index, self.current_file

    def data_generator(self):
        files = self.file_list
        samples = []
        batch_size = self.batch_size // self.max_seq_len if self.in_tokens \
                     else self.batch_size
        print("batch size: %d" % batch_size)
        self.current_file_index = -1
        self.total_file = len(files)
        self.current_file = "N/A"

        def local_iter():
            for f in files:
                samples = self.load_samples(f, self.shuffle)
                def reader():
                    for sample in samples:
                        yield sample
                batch_iter = paddle.batch(reader, batch_size)
                for batch in batch_iter():
                    yield self.convert_samples_to_fluid_tensors(batch)
                self.current_file_index += 1
                self.current_file = f
        return local_iter
