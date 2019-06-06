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
sys.path.append("..")
from tokenization import load_vocab
import tensorflow as tf
tf.enable_eager_execution()


class DataReader(object):
    def __init__(self,
                 data_dir,
                 vocab_path,
                 batch_size=4096,
                 in_tokens=True,
                 max_seq_len=128,
                 max_preds_per_seq=20,
                 shuffle_files=True,
                 is_test=False,
                 epoch=100,
                 **kwargs):
        self.vocab = load_vocab(vocab_path)
        self.pad_id = self.vocab["[PAD]"]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.epoch = epoch
        self.shuffle = shuffle_files
        self.in_tokens = in_tokens
        self.is_test = is_test
        self.name_to_features = {
            "input_ids": tf.FixedLenFeature([max_seq_len], tf.int64),
            "input_mask": tf.FixedLenFeature([max_seq_len], tf.int64),
            "segment_ids": tf.FixedLenFeature([max_seq_len], tf.int64),
            "masked_lm_positions":
            tf.FixedLenFeature([max_preds_per_seq], tf.int64),
            "masked_lm_ids": tf.FixedLenFeature([max_preds_per_seq], tf.int64),
            "masked_lm_weights":
            tf.FixedLenFeature([max_preds_per_seq], tf.float32),
            "next_sentence_labels": tf.FixedLenFeature([1], tf.int64)
        }

    def get_progress(self):
        """return current progress of traning data
        """
        return self.current_epoch, self.current_file_index, self.total_file, self.current_file

    def get_dataset(self, records, batch_size, num_cpu_threads=4):
        if not self.is_test:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(records))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(records))
            cycle_length = min(num_cpu_threads, len(records))
            d = d.apply(
                tf.data.experimental.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=True,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(records)
            d.repeat()

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: tf.parse_single_example(record, self.name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=False))

        return d

    def convert_to_fluid_tensors(self, tf_sample):
        src_ids = np.expand_dims(
            tf_sample["input_ids"], axis=-1).astype("int64")
        sent_ids = np.expand_dims(
            tf_sample["segment_ids"], axis=-1).astype("int64")
        labels = tf_sample["next_sentence_labels"].astype("int64")
        input_mask = tf_sample["input_mask"]

        batch_size = src_ids.shape[0]
        pos_ids = np.linspace(0, self.max_seq_len - 1, self.max_seq_len)
        pos_ids = np.tile(pos_ids, (batch_size, 1))
        pos_ids = input_mask * pos_ids

        input_mask = np.expand_dims(input_mask, axis=-1).astype("float32")
        pos_ids = np.expand_dims(pos_ids, axis=-1).astype("int64")

        masked_lm_weight = np.expand_dims(
            tf_sample["masked_lm_weights"], axis=-1)
        masked_lm_ids = np.expand_dims(tf_sample["masked_lm_ids"], axis=-1)
        masked_lm_pos = np.expand_dims(
            tf_sample["masked_lm_positions"], axis=-1)
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

    def data_generator(self):
        records = os.listdir(self.data_dir)
        records = [os.path.join(self.data_dir, record) for record in records]

        batch_size = self.batch_size // self.max_seq_len if self.in_tokens \
                     else self.batch_size
        self.current_file_index = -1
        self.total_file = len(records)
        self.current_file = "N/A"

        def wrapper():
            for epoch in xrange(self.epoch):
                self.current_epoch = epoch + 1
                if self.shuffle:
                    random.shuffle(records)
                data_set = self.get_dataset(records, batch_size)
                for data in iter(data_set):
                    tf_sample = {k: (v.numpy()) for k, v in data.items()}
                    yield self.convert_to_fluid_tensors(tf_sample)

        return wrapper
