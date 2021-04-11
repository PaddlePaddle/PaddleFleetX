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

import os
from propeller.service.server import InferenceProxy
from propeller.service.server import InferencePredictor
from propeller.service.server import run_worker

if __name__ == "__main__":
    model_dir = "/home/work/suweiyue/Release/infer_xnli/model/"
    n_devices = len(os.getenv("CUDA_VISIBLE_DEVICES").split(","))

    backend_addr = "tcp://10.255.129.36:5572"  # or  "tcp://localhost:5572"
    InferencePredictor(backend_addr, model_dir, n_devices).start()

    frontend_addr = "tcp://*:5571"
    backend_addr = "tcp://*:5572"
    InferenceProxy().listen(frontend_addr, backend_addr)
