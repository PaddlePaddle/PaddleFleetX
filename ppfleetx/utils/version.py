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

import paddle


def version_check():
    version = paddle.version.full_version
    if version != '0.0.0':
        paddle.utils.require_version(min_version='2.3.0')


def get_device_and_mapping():
    """
        Return gpu type and name-bool mapping implifying which type is supported.
    """
    suppoted_device_map = {
        "gpu": paddle.is_compiled_with_cuda(),
        "xpu": paddle.is_compiled_with_xpu(),
        "rocm": paddle.is_compiled_with_rocm(),
        "npu": paddle.is_compiled_with_npu(),
        "cpu": True
    }
    for d, v in suppoted_device_map.items():
        if v:
            return d, suppoted_device_map


def get_device():
    """
        Return the GPU Type, including 'gpu'(for rocm and gpu), 'npu', 'xpu', 'cpu'.
    """
    d, _ = get_device_and_mapping()
    return d

