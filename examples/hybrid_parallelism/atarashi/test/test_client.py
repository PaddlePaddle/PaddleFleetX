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
"""
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import time

import numpy as np
from propeller.service.client import InferenceClient
from propeller.service.client import InferenceBaseClient

if __name__ == "__main__":

    def line2nparray(line):
        slots = [slot.split(':') for slot in line.split(';')]
        dtypes = ["int64", "int64", "int64", "float32"]
        data_list = [
            np.reshape(
                np.array(
                    [float(num) for num in data.split(" ")], dtype=dtype),
                [int(s) for s in shape.split(" ")])
            for (shape, data), dtype in zip(slots, dtypes)
        ]
        return data_list

    #data_path = "/home/work/suweiyue/Release/infer_xnli/seq128_data/dev_ds"
    data_path = "/home/work/suweiyue/Share/model-compression/dev"
    address = "tcp://localhost:5575"
    client = InferenceClient(address, batch_size=50)

    data = []
    num = 10
    with open(data_path) as inf:
        for idx, line in enumerate(inf):
            if idx == num:
                break
            np_array = line2nparray(line.strip('\n'))
            data.append(np_array)

    begin = time.time()
    for np_array in data:
        ret = client(*np_array)
        print([r.shape for r in ret])
    print((time.time() - begin) / num)
