#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform

from setuptools import setup, Distribution, Extension
from setuptools import find_packages
from setuptools import setup
from fleet.version import paddle_fleet_version

def python_version():
    return [int(v) for v in platform.python_version().split(".")]

max_version, mid_version, min_version = python_version()

REQUIRED_PACKAGES = [
    'six >= 1.10.0', 'protobuf >= 3.1.0','paddlepaddle'
]

packages=['fleet',
          'fleet.proto',
          'fleet.io',
          'fleet.optimizer',
          'fleet.strategy',
          'fleet.utils']

package_dir={'fleet': 'fleet'}

setup(name='paddle-fleet',
      version=paddle_fleet_version.replace('-', ''),
      description=
      ('Large Scale Distributed Training Package For PaddlePaddle'),
      url='https://github.com/PaddlePaddle/Fleet',
      author='PaddlePaddle Author',
      author_email='guru4elephant@gmail.com',
      install_requires=REQUIRED_PACKAGES,
      packages=packages,
      package_data={'fleet': []},
      package_dir=package_dir,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      license='Apache 2.0',
      keywards=('paddle-fleet distributed training industrial solution easy-to-use'))
