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
import numpy as np
import io
from conf import *
import paddle.fluid.incubate.data_generator as dg

class NumpyRandomInt(object):
    def __init__(self, a, b, buf_size=1000):
        self.idx = 0
        self.buffer = np.random.random_integers(a, b, buf_size)
        self.a = a
        self.b = b

    def __call__(self):
        if self.idx == len(self.buffer):
            self.buffer = np.random.random_integers(self.a, self.b,
                                                    len(self.buffer))
            self.idx = 0

        result = self.buffer[self.idx]
        self.idx += 1
        return result

class MyDataset(dg.MultiSlotDataGenerator):
    def load_resource(self, dict_path, window_size, batch_size):
        self.batch_size = batch_size
        self.id_counts = []
        word_all_count = 0
        with io.open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word, count = line.split()[0], int(line.split()[1])
                self.id_counts.append(count)
                word_all_count += count
        self.id_frequencys = [
            float(count) / word_all_count for count in self.id_counts
        ]
        np_power = np.power(np.array(self.id_frequencys), 0.75)
        self.id_frequencys_pow = np_power / np_power.sum()
        self.dict_size = len(self.id_counts)
        self.random_generator = NumpyRandomInt(1, window_size + 1)

    def get_context_words(self, words, idx):
        target_window = self.random_generator()
        start_point = idx - target_window
        if start_point < 0:
            start_point = 0
        end_point = idx + target_window
        targets = words[start_point:idx] + words[idx + 1:end_point + 1]
        return targets
    
    def generate_sample(self, line):
        def data_iter():
            cs = np.array(self.id_frequencys_pow).cumsum()
            neg_array = cs.searchsorted(np.random.sample(neg_num))
            id_ = 0
            word_ids = [w for w in line.split()]
            for idx, target_id in enumerate(word_ids):
                context_word_ids = self.get_context_words(
                    word_ids, idx)
                for context_id in context_word_ids:
                    neg_id = [ int(str(i)) for i in neg_array ]
                    output = [('input_word', [int(target_id)]), ('true_label', [int(context_id)]), ('neg_label', neg_id)]
                    yield output
                    id_ += 1
                    if id_ % self.batch_size == 0:
                        neg_array = cs.searchsorted(np.random.sample(neg_num)) 
        return data_iter

if __name__ == "__main__":
    d = MyDataset()
    d.load_resource(dict_path, window_size, batch_size)
    d.run_from_stdin()
