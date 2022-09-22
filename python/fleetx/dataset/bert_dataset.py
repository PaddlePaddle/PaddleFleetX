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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import io
import random
import paddle
import six
import os
import gzip
import paddle.fluid as fluid
import numpy as np
import paddle.distributed.fleet as fleet


def load_bert_dataset(data_dir,
                      inputs,
                      batch_size,
                      lang='ch',
                      phase='train',
                      max_seq_len=512,
                      in_tokens=True,
                      shuffle=True,
                      generate_neg_sample=True,
                      voc_size=21128):
    data_loader = paddle.io.DataLoader.from_generator(
        feed_list=inputs, capacity=70, iterable=True)
    if lang == 'en':
        data_reader = DataReaderV2(
            data_dir=data_dir,
            phase=phase,
            batch_size=batch_size,
            in_tokens=in_tokens,
            max_seq_len=max_seq_len,
            shuffle_files=True)
    elif lang == 'ch':
        data_reader = DataReader(
            data_dir=data_dir,
            phase=phase,
            batch_size=batch_size,
            in_tokens=in_tokens,
            vocab_path="{}/vocab.txt".format(data_dir),
            voc_size=voc_size,
            epoch=1,
            max_seq_len=max_seq_len,
            generate_neg_sample=generate_neg_sample)

    else:
        raise Exception(
            "Bert Model only support chinese and english model now. Please check if your lang is one of 'ch' and 'en'"
        )
    places = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    data_loader.set_batch_generator(data_reader.data_generator(), places)
    return data_loader


class DataReader(object):
    def __init__(self,
                 data_dir,
                 vocab_path,
                 phase='train',
                 batch_size=4096,
                 in_tokens=True,
                 max_seq_len=512,
                 shuffle_files=True,
                 epoch=100,
                 voc_size=0,
                 is_test=False,
                 generate_neg_sample=False):

        self.vocab = self.load_vocab(vocab_path)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.in_tokens = in_tokens
        self.shuffle_files = shuffle_files
        self.epoch = epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.phase = phase
        self.total_file = 0
        self.current_file = None
        self.voc_size = voc_size
        self.max_seq_len = max_seq_len
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]
        self.is_test = is_test
        self.generate_neg_sample = generate_neg_sample
        if self.in_tokens:
            assert self.batch_size >= self.max_seq_len, "The number of " \
                   "tokens in batch should not be smaller than max seq length."

        if self.is_test:
            self.epoch = 1
            self.shuffle_files = False

    def get_progress(self):
        """return current progress of traning data
        """
        return self.current_epoch, self.current_file_index, self.total_file, self.current_file

    def parse_line(self, line, max_seq_len=512):
        """ parse one line to token_ids, sentence_ids, pos_ids, label
        """
        line = line.strip().decode().split(";")
        assert len(line) == 4, "One sample must have 4 fields!"
        (token_ids, sent_ids, pos_ids, label) = line
        token_ids = [int(token) for token in token_ids.split(" ")]
        sent_ids = [int(token) for token in sent_ids.split(" ")]
        pos_ids = [int(token) for token in pos_ids.split(" ")]
        assert len(token_ids) == len(sent_ids) == len(
            pos_ids
        ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"
        label = int(label)
        if len(token_ids) > max_seq_len:
            return None
        return [token_ids, sent_ids, pos_ids, label]

    def read_file(self, file):
        file_path = self.data_dir + "/" + file
        with open(file_path, "rb") as f:
            for line in f:
                parsed_line = self.parse_line(
                    line, max_seq_len=self.max_seq_len)
                if parsed_line is None:
                    continue
                yield parsed_line

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        #fin = io.open(vocab_file, encoding="utf8")
        fin = open(vocab_file)
        for num, line in enumerate(fin):
            items = self.convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    def random_pair_neg_samples(self, pos_samples):
        """ randomly generate negtive samples using pos_samples

            Args:
                pos_samples: list of positive samples

            Returns:
                neg_samples: list of negtive samples
        """
        np.random.shuffle(pos_samples)
        num_sample = len(pos_samples)
        neg_samples = []
        miss_num = 0

        for i in range(num_sample):
            pair_index = (i + 1) % num_sample
            origin_src_ids = pos_samples[i][0]
            origin_sep_index = origin_src_ids.index(2)
            pair_src_ids = pos_samples[pair_index][0]
            pair_sep_index = pair_src_ids.index(2)

            src_ids = origin_src_ids[:origin_sep_index + 1] + pair_src_ids[
                pair_sep_index + 1:]
            if len(src_ids) > self.max_seq_len:
                miss_num += 1
                continue
            sent_ids = [0] * len(origin_src_ids[:origin_sep_index + 1]) + [
                1
            ] * len(pair_src_ids[pair_sep_index + 1:])
            pos_ids = list(range(len(src_ids)))
            neg_sample = [src_ids, sent_ids, pos_ids, 0]
            assert len(src_ids) == len(sent_ids) == len(
                pos_ids
            ), "[ERROR]len(src_id) == lne(sent_id) == len(pos_id) must be True"
            neg_samples.append(neg_sample)
        return neg_samples, miss_num

    def mixin_negtive_samples(self, pos_sample_generator, buffer=1000):
        """ 1. generate negtive samples by randomly group sentence_1 and sentence_2 of positive samples
            2. combine negtive samples and positive samples

            Args:
                pos_sample_generator: a generator producing a parsed positive sample, which is a list: [token_ids, sent_ids, pos_ids, 1]

            Returns:
                sample: one sample from shuffled positive samples and negtive samples
        """
        pos_samples = []
        num_total_miss = 0
        pos_sample_num = 0
        try:
            while True:
                while len(pos_samples) < buffer:
                    pos_sample = next(pos_sample_generator)
                    label = pos_sample[3]
                    assert label == 1, "positive sample's label must be 1"
                    pos_samples.append(pos_sample)
                    pos_sample_num += 1

                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
        except StopIteration:
            print("stopiteration: reach end of file")
            if len(pos_samples) == 1:
                yield pos_samples[0]
            elif len(pos_samples) == 0:
                yield None
            else:
                neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
            print("miss_num:%d\tideal_total_sample_num:%d\tmiss_rate:%f" %
                  (num_total_miss, pos_sample_num * 2,
                   num_total_miss / (pos_sample_num * 2)))

    def data_generator(self):
        """
        data_generator
        """
        if self.phase == 'train':
            files = get_filelist(self.data_dir)
        else:
            files = get_val_filelist(self.data_dir)
        self.total_file = len(files)
        assert self.total_file > 0, "[Error] data_dir is empty"

        def wrapper():
            def reader():
                for epoch in range(self.epoch):
                    self.current_epoch = epoch + 1
                    if self.shuffle_files:
                        np.random.shuffle(files)
                    for index, file in enumerate(files):
                        self.current_file_index = index + 1
                        self.current_file = file
                        sample_generator = self.read_file(file)
                        if not self.is_test and self.generate_neg_sample:
                            sample_generator = self.mixin_negtive_samples(
                                sample_generator)
                        for sample in sample_generator:
                            if sample is None:
                                continue
                            yield sample

            def batch_reader(reader, batch_size, in_tokens):
                batch, total_token_num, max_len = [], 0, 0
                for parsed_line in reader():
                    token_ids, sent_ids, pos_ids, label = parsed_line
                    max_len = max(max_len, len(token_ids))
                    if in_tokens:
                        to_append = (len(batch) + 1) * max_len <= batch_size
                    else:
                        to_append = len(batch) < batch_size
                    if to_append:
                        batch.append(parsed_line)
                        total_token_num += len(token_ids)
                    else:
                        yield batch, total_token_num
                        batch, total_token_num, max_len = [parsed_line], len(
                            token_ids), len(token_ids)

                if len(batch) > 0:
                    yield batch, total_token_num

            for batch_data, total_token_num in batch_reader(
                    reader, self.batch_size, self.in_tokens):
                yield prepare_batch_data(
                    batch_data,
                    total_token_num,
                    voc_size=self.voc_size,
                    pad_id=self.pad_id,
                    cls_id=self.cls_id,
                    sep_id=self.sep_id,
                    mask_id=self.mask_id,
                    return_input_mask=True,
                    return_max_len=False,
                    return_num_token=False)

        return wrapper


class DataReaderV2(object):
    def __init__(self,
                 data_dir,
                 phase='train',
                 batch_size=4096,
                 in_tokens=True,
                 max_seq_len=128,
                 max_preds_per_seq=20,
                 shuffle_files=True,
                 is_test=False,
                 epoch=100,
                 **kwargs):
        self.data_dir = data_dir
        if phase == 'train':
            file_list = get_filelist(data_dir)
        else:
            file_list = get_val_filelist(data_dir)
        self.file_list = [data_dir + "/" + x for x in file_list]
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.epoch = epoch
        self.shuffle = shuffle_files
        self.in_tokens = in_tokens
        self.is_test = is_test
        self.current_epoch = -1
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None

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
            input_mask.extend(to_int_array(group[1]))
            sent_ids.extend(to_int_array(group[2]))
            masked_lm_pos.extend(to_int_array(group[3]))
            masked_lm_ids.extend(to_int_array(group[4]))
            masked_lm_weight.extend(to_float_array(group[5]))
            labels.extend(to_int_array(group[6]))

        src_ids = np.array(src_ids).reshape(batch_size, self.max_seq_len)
        sent_ids = np.array(sent_ids).reshape(batch_size, self.max_seq_len)
        input_mask = \
                np.array(input_mask).astype("int64").reshape(
                    batch_size, self.max_seq_len)
        labels = np.array(labels).astype("int64").reshape(batch_size, 1)
        pos_ids = (input_mask * pos_ids).astype("int64")
        input_mask = np.expand_dims(input_mask, axis=-1).astype("float32")

        masked_lm_weight = np.array(masked_lm_weight).reshape(
            batch_size, -1).astype("float32")
        masked_lm_ids = np.array(masked_lm_ids).reshape(batch_size,
                                                        -1).astype("int64")
        masked_lm_pos = np.array(masked_lm_pos).reshape(batch_size,
                                                        -1).astype("int64")

        masked_cnt = masked_lm_weight.sum(axis=1).astype("int").flatten()
        mask_label = np.empty((1, 0))
        mask_pos = np.empty((1, 0))
        for i, cnt in enumerate(masked_cnt):
            mask_label = np.append(mask_label, masked_lm_ids[i, 0:cnt])
            mask_pos = np.append(
                mask_pos, masked_lm_pos[i, 0:cnt] + i * self.max_seq_len)

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

    def data_generator(self):
        files = self.file_list
        samples = []
        batch_size = self.batch_size // self.max_seq_len if self.in_tokens \
                     else self.batch_size
        print("batch size: %d" % batch_size)
        self.total_file = len(files)
        self.current_file = "N/A"

        def local_iter():
            for epoch in range(self.epoch):
                self.current_epoch = epoch
                self.current_file_index = -1
                for index, f in enumerate(files):
                    self.current_file_index = index + 1
                    self.current_file = f
                    samples = self.load_samples(f, self.shuffle)

                    def reader():
                        for sample in samples:
                            yield sample

                    batch_iter = paddle.batch(reader, batch_size)
                    for batch in batch_iter():
                        converted_samples = self.convert_samples_to_fluid_tensors(
                            batch)
                        # print(converted_samples[0].shape)
                        yield converted_samples

        return local_iter

    def get_progress(self):
        return self.current_epoch, self.current_file_index, self.total_file, self.current_file


def prepare_batch_data(insts,
                       total_token_num,
                       voc_size=0,
                       pad_id=None,
                       cls_id=None,
                       sep_id=None,
                       mask_id=None,
                       return_input_mask=True,
                       return_max_len=True,
                       return_num_token=False):
    """
    1. generate Tensor of data
    2. generate Tensor of position
    3. generate self attention mask, [shape: batch_size *  max_len * max_len]
    """

    batch_src_ids = [inst[0] for inst in insts]
    batch_sent_ids = [inst[1] for inst in insts]
    batch_pos_ids = [inst[2] for inst in insts]
    labels_list = []
    # compatible with squad, whose example includes start/end positions,
    # or unique id

    for i in range(3, len(insts[0]), 1):
        labels = [inst[i] for inst in insts]
        labels = np.array(labels).astype("int64").reshape([-1, 1])
        labels_list.append(labels)

    # First step: do mask without padding
    if mask_id >= 0:
        out, mask_label, mask_pos = mask(
            batch_src_ids,
            total_token_num,
            vocab_size=voc_size,
            CLS=cls_id,
            SEP=sep_id,
            MASK=mask_id)
    else:
        out = batch_src_ids
    # Second step: padding
    src_id, self_input_mask = pad_batch_data(
        out, pad_idx=pad_id, return_input_mask=True)
    pos_id = pad_batch_data(
        batch_pos_ids,
        pad_idx=pad_id,
        return_pos=False,
        return_input_mask=False)
    sent_id = pad_batch_data(
        batch_sent_ids,
        pad_idx=pad_id,
        return_pos=False,
        return_input_mask=False)

    if mask_id >= 0:
        return_list = [
            src_id, pos_id, sent_id, self_input_mask, mask_label, mask_pos
        ] + labels_list
    else:
        return_list = [src_id, pos_id, sent_id, self_input_mask] + labels_list

    return return_list if len(return_list) > 1 else return_list[0]


def pad_batch_data(insts,
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and input mask.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array([
        list(inst) + list([pad_idx] * (max_len - len(inst))) for inst in insts
    ])
    return_list += [inst_data.astype("int64").reshape([-1, max_len])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] *
                                    (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    return return_list if len(return_list) > 1 else return_list[0]


def mask(batch_tokens, total_token_num, vocab_size, CLS=1, SEP=2, MASK=3):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    max_len = max([len(sent) for sent in batch_tokens])
    mask_label = []
    mask_pos = []
    prob_mask = np.random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    replace_ids = np.random.randint(1, high=vocab_size, size=total_token_num)
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        mask_flag = False
        prob_index += pre_sent_len
        for token_index, token in enumerate(sent):
            prob = prob_mask[prob_index + token_index]
            if prob > 0.15:
                continue
            elif 0.03 < prob <= 0.15:
                # mask
                if token != SEP and token != CLS:
                    mask_label.append(sent[token_index])
                    sent[token_index] = MASK
                    mask_flag = True
                    mask_pos.append(sent_index * max_len + token_index)
            elif 0.015 < prob <= 0.03:
                # random replace
                if token != SEP and token != CLS:
                    mask_label.append(sent[token_index])
                    sent[token_index] = replace_ids[prob_index + token_index]
                    mask_flag = True
                    mask_pos.append(sent_index * max_len + token_index)
            else:
                # keep the original token
                if token != SEP and token != CLS:
                    mask_label.append(sent[token_index])
                    mask_pos.append(sent_index * max_len + token_index)
        pre_sent_len = len(sent)

        while not mask_flag:
            token_index = int(np.random.randint(1, high=len(sent) - 1, size=1))
            if sent[token_index] != SEP and sent[token_index] != CLS:
                mask_label.append(sent[token_index])
                sent[token_index] = MASK
                mask_flag = True
                mask_pos.append(sent_index * max_len + token_index)
    mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
    return batch_tokens, mask_label, mask_pos


def get_filelist(datadir):
    if not os.path.exists("{}/train.txt".format(datadir)):
        raise Exception("ERROR: Your data dir should include train.txt")

    total_list = []
    full_list = []
    with open("{}/train.txt".format(datadir), 'r') as fin:
        for line in fin:
            current_file = line.strip()
            if os.path.exists("{}/{}".format(datadir, current_file)):
                total_list.append(current_file)
            full_list.append(current_file)
    total_num = len(total_list)
    if len(total_list) == len(full_list):
        print("files to train on this card: {}".format(total_list[
            fleet.worker_index()::fleet.worker_num()]))
        return total_list[fleet.worker_index()::fleet.worker_num()]
    PADDLE_TRAINER_ENDPOINTS = os.environ.get('PADDLE_TRAINER_ENDPOINTS')
    current_endpoint = os.environ.get('PADDLE_CURRENT_ENDPOINT')
    endpoints = PADDLE_TRAINER_ENDPOINTS.split(",")
    hostname, _ = current_endpoint.split(":")
    host_endpoints = [x for x in endpoints if x.split(":")[0] == hostname]
    current_id = host_endpoints.index(current_endpoint)
    total_local_cards = len(host_endpoints)
    print("files to train on this card: {}".format(total_list[
        current_id::total_local_cards]))
    return total_list[current_id::total_local_cards]


def get_val_filelist(datadir):
    if not os.path.exists("{}/val.txt".format(datadir)):
        raise Exception("ERROR: Your data dir should include val.txt")

    total_list = []
    with open("{}/val.txt".format(datadir), 'r') as fin:
        for line in fin:
            current_file = line.strip()
            if os.path.exists("{}/{}".format(datadir, current_file)):
                total_list.append(current_file)

    return total_list
