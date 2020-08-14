# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import fleetx as X
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import time
# FleetX help users to focus more on learning to train a large scale model
# if you want to learn how to write a model, FleetX is not for you
# focus more on engineering staff in fleet-x

if not os.path.exists('testdata.tar.gz'):
    print("============download data============")
    os.system(
        'wget --no-check-certificate https://fleet.bj.bcebos.com/models/testdata.tar.gz'
    )
    os.system('tar -xf testdata.tar.gz')

configs = X.parse_train_configs()
os.environ['FLAGS_selected_gpus'] = '0'
model = X.applications.Resnet50()
test_program = fluid.default_main_program().clone(for_test=True)
loader = model.load_imagenet_from_file("./testdata/train.txt")
test_loader = model.load_imagenet_from_file("./testdata/val.txt", phase='val')

optimizer = fluid.optimizer.Momentum(
    learning_rate=configs.lr,
    momentum=configs.momentum,
    regularization=fluid.regularizer.L2Decay(0.0001))
optimizer.minimize(model.loss, parameter_list=model.parameter_list())

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
model_dir = 'model'
print("============start training============")
for epoch_id in range(2):
    total_time = 0
    for i, data in enumerate(loader()):
        if i >= 10:
            start_time = time.time()
        cost_val = exe.run(fluid.default_main_program(),
                           feed=data,
                           fetch_list=[model.loss.name],
                           use_program_cache=True)
        if i >= 10:
            end_time = time.time()
            total_time += (end_time - start_time)
            print(
                "epoch%d step%d cost = %f, total time cost = %f, average speed = %f"
                % (epoch_id, i, cost_val[0], total_time, (i - 9) / total_time))
    fluid.io.save_inference_model(
        dirname=model_dir,
        feeded_var_names=[model.inputs[0].name],
        target_vars=model.target,
        executor=exe)

print("============start inference============")
for j, test_data in enumerate(test_loader()):
    acc1, acc5 = exe.run(test_program,
                         feed=test_data,
                         fetch_list=[t.name for t in model.target],
                         use_program_cache=True)
    print("acc1 = %f, acc5 = %f" % (acc1[0], acc5[0]))
