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
import  random

def combination(x, y):
    res = [[[xi, yi] for yi in y] for xi in x]
    return res[0]


def get_one_data(file_list, sample_rate):
    for file in file_list:
        contents = []
        with open(file, "r") as fin:
            for q in fin.readlines():
                """query_ids, pos_title_ids, neg_title_ids, label"""

                one_data = q.split(";")[:-1]

                if len(one_data) < 4:
                    print("data format error!, please check!", q)
                    continue

                label = int(one_data[0])
                pos_title_num, neg_title_num = int(one_data[1].split(" ")[0]), int(one_data[1].split(" ")[1])
                query_ids = [int(x) for x in one_data[2].split(" ")]

                if pos_title_num + neg_title_num != len(one_data) - 3:
                    print("data format error, pos_title_num={}, neg_title_num={}, one_data={}"
                                .format(pos_title_num, neg_title_num, len(one_data)))
                    continue

                for x in range(pos_title_num):
                    pos_title_ids = [ int(i) for i in one_data[3+x].split(" ")]
                    for y in range(neg_title_num):
                        if random.random() > sample_rate:
                            continue
                        neg_title_ids = [int(i) for i in one_data[3+pos_title_num+y].split(" ")]
                        yield [query_ids, pos_title_ids, neg_title_ids, [label]]
        fin.close()

def get_batch_reader(file_list, batch_size=128, sample_rate=0.02, trainer_id=1):
    def batch_reader():
        res = []
        idx = 0
        for i in get_one_data(file_list, sample_rate):
            res.append(i)
            idx += 1
            if len(res) >= batch_size:
                yield res
                res = []
    return batch_reader

def get_infer_data(file_list, sample_rate):
    for file in file_list:
        contents = []
        with open(file, "r") as fin:
            for q in fin.readlines():
                """query_ids, pos_title_ids, neg_title_ids, label"""

                one_data = q.split(";")[:-1]

                if len(one_data) < 4:
                    print("data format error!, please check!",q)
                    continue

                label = int(one_data[0])
                pos_title_num, neg_title_num = int(one_data[1].split(" ")[0]), int(one_data[1].split(" ")[1])
                query_ids = [int(x) for x in one_data[2].split(" ")]

                if pos_title_num + neg_title_num != len(one_data) - 3:
                    print("data format error, pos_title_num={}, neg_title_num={}, one_data={}"
                                .format(pos_title_num,neg_title_num,len(one_data)))
                    continue

                for x in range(pos_title_num):
                    pos_title_ids = [int(i) for i in one_data[3 + x].split(" ")]
                    for y in range(neg_title_num):
                        if random.random() > sample_rate:
                            continue
                        neg_title_ids = [int(i) for i in one_data[3 + pos_title_num + y].split(" ")]
                        yield [query_ids, pos_title_ids, neg_title_ids]
        fin.close()

def get_infer_batch_reader(file_list, batch_size=128, sample_rate=0.02, trainer_id=1):
    def batch_reader():
        res = []
        idx = 0
        for i in get_infer_data(file_list, sample_rate):
            res.append(i)
            idx += 1
            if len(res) >= batch_size:
                yield res
                res = []
    return batch_reader

