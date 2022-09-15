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
import csv

import paddle

from ppfleetx.data.tokenizers import GPTTokenizer

__all__ = ['SST2', ]


def parse_csv(path, skip_lines=0, delimiter=' ', quotechar='|', func=None):

    with open(path, newline='') as csvfile:
        data = []
        spamreader = csv.reader(
            csvfile, delimiter=delimiter, quotechar=quotechar)
        for idx, row in enumerate(spamreader):
            if idx < skip_lines:
                continue
            if func is not None:
                row = func(row)
            data.append(row)
        return data


class SST2(paddle.io.Dataset):

    # ref https://pytorch.org/text/stable/_modules/torchtext/datasets/sst2.html#SST2

    URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
    MD5 = "9f81648d4199384278b86e315dac217c"

    NUM_LINES = {
        "train": 67349,
        "dev": 872,
        "test": 1821,
    }

    _PATH = "SST-2.zip"

    DATASET_NAME = "SST2"

    _EXTRACTED_FILES = {
        "train": os.path.join("SST-2", "train.tsv"),
        "dev": os.path.join("SST-2", "dev.tsv"),
        "test": os.path.join("SST-2", "test.tsv"),
    }

    def __init__(self, root, split):

        self.root = root
        self.split = split
        self.path = os.path.join(self.root, self._EXTRACTED_FILES[split])
        self.max_seq_length = max_seq_length

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        assert split in ['train', 'dev', 'test']

        # test split for SST2 doesn't have labels
        if split == "test":

            def _modify_test_res(t):
                return (t[1].strip(), )

            self.samples = parse_csv(
                self.path, skip_lines=1, delimiter="\t", func=_modify_test_res)
        else:

            def _modify_res(t):
                return t[0].strip(), int(t[1])

            self.samples = parse_csv(
                self.path, skip_lines=1, delimiter="\t", func=_modify_res)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = self.tokenizer.encode(sample[0])
        # TODO(GuoxiaWang): add padding and truncate to max_seq_length

        if self.split != 'test':
            return input_ids, sample[1]
        else:
            return input_ids

    def __len__(self):
        return len(self.samples)

    @property
    def class_num(self):
        return 2
