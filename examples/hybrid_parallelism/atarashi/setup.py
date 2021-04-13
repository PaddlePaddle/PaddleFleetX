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
"""
doc
"""

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, find_packages


def get_dist(pkgname):
    """doc"""
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


install_deps = []

if get_dist('paddlepaddle') is None and get_dist('paddlepaddle_gpu') is None:
    install_deps.append('paddlepaddle')

setup(
    name='paddle-propeller',
    version='0.3dev1',
    description='high level paddle-paddle API',
    url='https://github.com/PaddlePaddle/ERNIE',
    author='Chen Xuyi',
    author_email='chen_xuyi@outlook.com',
    license='Apache 2.0',
    packages=find_packages(),
    python_requires='>= 2.6.*',
    install_requires=install_deps)
