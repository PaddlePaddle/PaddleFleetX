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

import os
import csv
import json
import random
import numpy as np
import math
from collections import namedtuple

import utils.tokenization as tokenization
from utils.batching import pad_batch_data


class BaseReader(object):
    def __init__(self,
                 trainer_id,
                 trainer_num,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_classify=True,
                 is_regression=False,
                 text_field_more_than_3=False):

        self.trainer_id = trainer_id
        self.trainer_num = trainer_num
        self.text_field_more_than_3 = text_field_more_than_3

        if random_seed is None:
            random_seed = 12345
        self.rng = random.Random(random_seed)

        self.max_seq_len = max_seq_len
        #self.tokenizer = getattr(tokenization, tokenizer)(
        #        vocab_file=vocab_path, do_lower_case=do_lower_case)
        #self.tokenizer = tokenization.BPETokenizer('./package/vocab.txt', './package/encoder.json', './package/vocab.bpe')
        self.tokenizer = tokenization.FullTokenizer(None, spm_model_file='./config/30k-clean.model')
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["<pad>"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]

        self.in_tokens = in_tokens

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.is_classify = is_classify
        self.is_regression = is_regression

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                example = Example(*line)
                examples.append(example)
            return examples

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _truncate_seqs(self, tokens_of_sub_sentence, max_num_tokens):
        while True:
            ls = [len(ts) for ts in tokens_of_sub_sentence]
            total_length = sum(ls)
            if total_length <= max_num_tokens:
                break
            max_l = max(ls)
            ind = ls.index(max_l)
            trunc_tokens = tokens_of_sub_sentence[ind]

            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if self.rng.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def get_all_text_field(self, example):
        values = []
        for i in range(ord("a"),ord("z")):
            field_name = 'text_' + chr(i)

            if isinstance(example, dict):
                has_field = field_name in example.keys()
            else:
                has_field = field_name in example._fields

            if has_field:
                v = getattr(example, field_name)
                values.append(v)
        return values

    def _convert_example_to_record_3(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""
        values = self.get_all_text_field(example)

        all_tokens = []
        for text in values:
            text_a = tokenization.convert_to_unicode(text)
            tokens = tokenizer.tokenize(text_a)
            all_tokens.append(tokens)

        self._truncate_seqs(all_tokens, max_seq_length - len(all_tokens))

        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for i, _tokens in enumerate(all_tokens):
            for token in _tokens:
                tokens.append(token)
                text_type_ids.append(i)
            text_type_ids.append(i)
            tokens.append("[SEP]")

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        task_ids = [0] * len(token_ids)

        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'label_id', 'qid'
        ])

        qid = None
        if "qid" in example._fields:
            qid = example.qid

        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            label_id=label_id,
            qid=qid)
        return record

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""

        text_a = example.text_a
        text_a = tokenization.preprocess_text(text_a, lower=True)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None

        has_text_b = False
        if isinstance(example, dict):
            has_text_b = "text_b" in example.keys()
        else:
            has_text_b = "text_b" in example._fields

        if has_text_b:
            text_b = example.text_b
            text_b = tokenization.preprocess_text(text_b, lower=True)
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0: (max_seq_length - 2)]

        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append("[SEP]")
        text_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'label_id', 'qid'])

        qid = None
        if "qid" in example._fields:
            qid = example.qid

        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            label_id=label_id,
            qid=qid)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index

            if self.text_field_more_than_3:
                record = self._convert_example_to_record_3(example, self.max_seq_len,
                                                         self.tokenizer)
            else:
                record = self._convert_example_to_record(example, self.max_seq_len,
                                                         self.tokenizer)
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        # apply drop last to avoid hanging in training
        if self.trainer_num == 1 or (phase != 'train' and batch_records):
            yield self._pad_batch_records(batch_records)

    def get_num_examples(self, input_file):
        examples = self._read_tsv(input_file)
        return len(examples)

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       shuffle=True,
                       phase=None):
        examples = self._read_tsv(input_file)

        def wrapper():
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    self.global_rng = np.random.RandomState(self.random_seed)
                    self.global_rng.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase):
                    if len(all_dev_batches) < self.trainer_num:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == self.trainer_num:
                        yield all_dev_batches[self.trainer_id]
                        all_dev_batches = []

            if phase != 'train' and len(all_dev_batches) > 0:
                last_batch = all_dev_batches[-1]
                for i in range(len(last_batch)):
                    lbl = last_batch[i].tolist()
                    while len(lbl) < batch_size:
                        lbl.append(lbl[0])
                    last_batch[i] = np.array(lbl)
                all_dev_batches[-1] = last_batch

                while len(all_dev_batches) < self.trainer_num:
                    all_dev_batches.append(all_dev_batches[-1])
                yield all_dev_batches[self.trainer_id]

        return wrapper


class ClassifyReader(BaseReader):

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            label_indices = [
                index for index, h in enumerate(headers) if h == "label"
            ]

            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                example = Example(*line)
                examples.append(example)
            return examples

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_labels = [record.label_id for record in batch_records]

        if self.is_classify:
            batch_labels = np.array(batch_labels).astype("int64").reshape([-1, 1])
        elif self.is_regression:
            batch_labels = np.array(batch_labels).astype("float32").reshape([-1, 1])

        if batch_records[0].qid:
            batch_qids = [record.qid for record in batch_records]
            batch_qids = np.array(batch_qids).astype("int64").reshape([-1, 1])
        else:
            batch_qids = np.array([]).astype("int64").reshape([-1, 1])

        # padding
        padded_token_ids = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, max_seq_len=self.max_seq_len)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id, max_seq_len=self.max_seq_len)

        return_list = [
            padded_token_ids, padded_text_type_ids, batch_labels, batch_qids,
        ]

        return return_list


class SequenceLabelReader(BaseReader):
    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_label_ids = [record.label_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_label_ids = pad_batch_data(
            batch_label_ids, pad_idx=len(self.label_map) - 1)
        padded_task_ids = pad_batch_data(
            batch_task_ids, pad_idx=self.pad_id)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            input_mask, padded_label_ids, batch_seq_lens
        ]
        return return_list

    def _reseg_token_label(self, tokens, labels, tokenizer):
        assert len(tokens) == len(labels)
        ret_tokens = []
        ret_labels = []
        for token, label in zip(tokens, labels):
            sub_token = tokenizer.tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            ret_labels.append(label)
            if len(sub_token) < 2:
                continue
            sub_label = label
            if label.startswith("B-"):
                sub_label = "I-" + label[2:]
            ret_labels.extend([sub_label] * (len(sub_token) - 1))

        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        tokens = tokenization.convert_to_unicode(example.text_a).split(u"")
        labels = tokenization.convert_to_unicode(example.label).split(u"")
        tokens, labels = self._reseg_token_label(tokens, labels, tokenizer)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))
        text_type_ids = [0] * len(token_ids)
        no_entity_id = len(self.label_map) - 1
        label_ids = [no_entity_id] + [
            self.label_map[label] for label in labels
        ] + [no_entity_id]

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_ids'])
        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            label_ids=label_ids)
        return record


class ExtractEmbeddingReader(BaseReader):
    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)

        assert(len(padded_token_ids[0]) == 128)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            input_mask, seq_lens
        ]

        return return_list


if __name__ == '__main__':
    pass
