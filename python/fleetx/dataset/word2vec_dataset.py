# -*- coding: utf-8 -*
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
import logging
import io
import os
import six
import paddle
import paddle.fluid as fluid

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


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


class Word2VecReader(object):
    def __init__(self,
                 dict_path,
                 filelist,
                 window_size=5):
        self.window_size_ = window_size
        self.filelist = filelist

        word_all_count = 0
        id_counts = []
        word_id = 0

        with io.open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word, count = line.split()[0], int(line.split()[1])
                word_id += 1
                id_counts.append(count)
                word_all_count += count

        self.word_all_count = word_all_count
        self.corpus_size_ = word_all_count
        self.dict_size = len(id_counts)
        self.id_counts_ = id_counts

        print("corpus_size:", self.corpus_size_)
        self.id_frequencys = [
            float(count) / word_all_count for count in self.id_counts_
        ]
        print("dict_size = " + str(self.dict_size) + " word_all_count = " + str(
            word_all_count))

        self.random_generator = NumpyRandomInt(1, self.window_size_ + 1)

    def get_context_words(self, words, idx):
        """
        Get the context word list of target word.
        words: the words of the current line
        idx: input word index
        window_size: window size
        """
        target_window = self.random_generator()
        start_point = idx - target_window  # if (idx - target_window) > 0 else 0
        if start_point < 0:
            start_point = 0
        end_point = idx + target_window
        targets = words[start_point:idx] + words[idx + 1:end_point + 1]
        return targets

    def train(self):
        def nce_reader():
            for file in self.filelist:
                with io.open(file, 'r', encoding='utf-8') as f:
                    logger.info("running data in {}".format(file))
                    count = 1
                    for line in f:
                        word_ids = [int(w) for w in line.split()]
                        for idx, target_id in enumerate(word_ids):
                            context_word_ids = self.get_context_words(
                                word_ids, idx)
                            for context_id in context_word_ids:
                                yield [target_id], [context_id]
                        count += 1
                        if count >= 20000:
                            break

        return nce_reader


def convert_python_to_numpy(weight, batch_size, sample_reader, nce_num):
    def __reader__():
        cs = np.array(weight).cumsum()
        result = [[], []]
        for sample in sample_reader():
            for i, fea in enumerate(sample):
                result[i].append(fea)
            if len(result[0]) == batch_size:
                tensor_result = []
                for tensor in result:
                    dat = np.array(tensor, dtype='int64')
                    if len(dat.shape) > 2:
                        dat = dat.reshape((dat.shape[0], dat.shape[2]))
                    elif len(dat.shape) == 1:
                        dat = dat.reshape((-1, 1))
                    tensor_result.append(dat)
                neg_array = cs.searchsorted(np.random.sample(nce_num))
                neg_array = np.tile(neg_array, batch_size)
                tensor_result.append(neg_array.reshape((batch_size, nce_num)))
                yield tensor_result
                result = [[], []]

    return __reader__

def BuildWord_IdMap(dict_path):
    word_to_id = dict()
    id_to_word = dict()

    with open(dict_path, 'r') as f:
        print("dict path : %s"%dict_path)
        for line in f:
            word_to_id[line.split(' ')[0]] = int(line.split(' ')[1])
            id_to_word[int(line.split(' ')[1])] = line.split(' ')[0]
    return word_to_id, id_to_word


def prepare_data(file_dir, dict_path, batch_size):
    w2i, i2w = BuildWord_IdMap(dict_path)
    vocab_size = len(i2w)
    reader = paddle.batch(test(file_dir, w2i), batch_size)
    return vocab_size, reader, i2w


def native_to_unicode(s):
    if _is_unicode(s):
        return s
    try:
        return _to_unicode(s)
    except UnicodeDecodeError:
        res = _to_unicode(s, ignore_errors=True)
        return res


def _is_unicode(s):
    if six.PY2:
        if isinstance(s, unicode):
            return True
    else:
        if isinstance(s, str):
            return True
    return False


def _to_unicode(s, ignore_errors=False):
    if _is_unicode(s):
        return s
    error_mode = "ignore" if ignore_errors else "strict"
    return s.decode("utf-8", errors=error_mode)


def strip_lines(line, vocab):
    return _replace_oov(vocab, native_to_unicode(line))


def _replace_oov(original_vocab, line):
    """Replace out-of-vocab words with "<UNK>".
  This maintains compatibility with published results.
  Args:
    original_vocab: a set of strings (The standard vocabulary for the dataset)
    line: a unicode string - a space-delimited sequence of words.
  Returns:
    a unicode string - a space-delimited sequence of words.
  """
    return u" ".join([
        word if word in original_vocab else u"<UNK>" for word in line.split()
    ])


def reader_creator(file_dir, word_to_id):
    def reader():
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + '/' + fi, "r") as f:
                for line in f:
                    if ':' in line:
                        pass
                    else:
                        line = strip_lines(line.lower(), word_to_id)
                        line = line.split()
                        yield [word_to_id[line[0]]], [word_to_id[line[1]]], [
                            word_to_id[line[2]]
                        ], [word_to_id[line[3]]], [
                            word_to_id[line[0]], word_to_id[line[1]],
                            word_to_id[line[2]]
                        ]

    return reader


def load_w2v_dataset(inputs,
            train_files_path,
            dict_path,
            nce_num,
            batch_size,
            shuffle):
    data_loader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=70, iterable=True)
    word2vec_reader = Word2VecReader(dict_path, train_files_path)
    np_power = np.power(np.array(word2vec_reader.id_frequencys), 0.75)
    id_frequencys_pow = np_power / np_power.sum()

    places = fluid.CPUPlace()
    data_loader.set_batch_generator(convert_python_to_numpy(id_frequencys_pow, batch_size, word2vec_reader.train(), nce_num), places)
    return data_loader


def test(test_dir, w2i):
    return reader_creator(test_dir, w2i)
