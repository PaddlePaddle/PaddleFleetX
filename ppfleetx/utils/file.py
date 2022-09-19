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
import zipfile
import paddle
from ppfleetx.utils.env import work_at_local_rank0


@work_at_local_rank0
def unzip(zip_path, mode="r", out_dir=None, delete=False):
    with zipfile.ZipFile(zip_path, mode) as zip_ref:
        zip_ref.extractall(out_dir)

    if delete:
        os.remove(zip_path)


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
