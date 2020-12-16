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
import paddle
import fleetx as X
import paddle.fluid as fluid


paddle.enable_static()
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
loader = model.get_train_dataloader("./testdata")
test_loader = model.get_val_dataloader("./testdata")
print(model.target)
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
trainer = X.MultiGPUTrainer()
trainer.fit(model, loader, epoch=1)
fluid.io.save_inference_model(
    dirname=model_dir,
    feeded_var_names=[model.inputs[0].name],
    target_vars=[model.target['acc1'], model.target['acc5']],
    executor=exe)

print("============start inference============")
trainer.val(model, test_loader, target_list=['acc1', 'acc5'])
