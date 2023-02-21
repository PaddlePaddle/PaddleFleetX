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

from setuptools import setup, Extension, find_packages

from ppfleetx.data.data_tools.cpp.compile import compile_helper
compile_helper()


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


install_requires = fetch_requirements('requirements.txt')

setup(
    name='ppfleetx',
    version='0.0.0',
    description='PaddleFleetX',
    author='PaddlePaddle Authors',
    url='https://github.com/PaddlePaddle/PaddleFleetX',
    install_requires=install_requires,
    package_data={
        'ppfleetx.data.data_tools.cpp': ['fast_index_map_helpers.so']
    },
    packages=find_packages())
