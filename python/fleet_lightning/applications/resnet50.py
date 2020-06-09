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
from .util import *
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy

class ModelBase(object):
    def __init__(self):
        self.inputs = []
        self.startup_prog = None
        self.main_prog = None

    def inputs(self):
        return self.inputs

    def loss(self):
        return self.loss

    def hidden(self):
        return []

    def parameter_list(self):
        return self.main_prog.all_parameters()

    def startup_program(self):
        return self.startup_prog

    def main_program(self):
        return self.main_prog
    

class Resnet50(ModelBase):
    def __init__(self):
        super(Resnet50, self).__init__()
        inputs, loss, startup, main, unique_generator = load_program("resnet50")
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss

