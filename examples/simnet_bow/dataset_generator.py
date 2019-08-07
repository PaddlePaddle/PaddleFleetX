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

from __future__ import print_function
import paddle.fluid.incubate.data_generator as dg
import random

class PairwiseReader(dg.MultiSlotDataGenerator):
    def init_reader(self, max_len, sampling_rate):
        # np.random.seed(1)
        self.max_len = max_len
        self.sampling_rate = sampling_rate
        self.query_buffer = None
        self.pos_title_buffer = None
        self.neg_title_buffer = None

    def infer_reader(self, infer_filelist, batch, buf_size):
        def local_iter():
            for fname in infer_filelist:
                with open(fname, "r") as fin:
                    for line in fin:
                        items = line.strip("\t\n").split(";")
                        pos_num, neg_num = [int(i) for i in items[1].split(" ")]
                        query = [int(j) for j in items[2].split(" ")]
                        for i in range(pos_num):
                            for j in range(neg_num):
                                pos_title_int = [int(x) for x in items[3 + i].split(" ")]
                                neg_title_int = [int(x) for x in items[3 + pos_num + j].split(" ")]
                                yield query, pos_title_int, neg_title_int

        import paddle
        batch_iter = paddle.batch(
            paddle.reader.shuffle(local_iter, buf_size=buf_size),
            batch_size=batch)
        return batch_iter

    def generate_sample(self, line):
        def get_rand(low=0.0, high=1.0):
            return random.random()

        def pairwise_iterator():
            items = line.strip("\t\n").split(";")
            pos_num, neg_num = [int(i) for i in items[1].split(" ")]
            query = [int(j) for j in items[2].split(" ")]

            for i in range(pos_num):
                for j in range(neg_num):
                    prob = get_rand()
                    if prob < self.sampling_rate:
                        pos_title_int = [int(x) for x in items[3 + i].split(" ")]
                        neg_title_int = [int(x) for x in items[3 + pos_num + j].split(" ")]
                        yield ("query", query), \
                              ("pos_title", pos_title_int), \
                              ("neg_title", neg_title_int)

        return pairwise_iterator


if __name__ == "__main__":
    pairwise_reader = PairwiseReader()
    pairwise_reader.init_reader(10000, 0.02)
    pairwise_reader.run_from_stdin()
