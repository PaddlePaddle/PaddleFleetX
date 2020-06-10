import os
import fleet_lightning as lighting
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
# lightning help users to focus more on learning to train a large scale model
# if you want to learn how to write a model, lightning is not for you
# focus more on engineering staff in fleet-lightning


configs = lighting.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)

model = lighting.applications.Resnet50()
loader = lighting.image_dataset_from_filelist(
    "/ssd2/lilong/ImageNet/train.txt", configs)

optimizer = fluid.optimizer.Momentum(
    learning_rate=configs.lr,
    momentum=configs.momentum,
    parameter_list=model.parameter_list(),
    regularization=fluid.regularizer.L2Decay(0.0001))
optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(model.loss,
                   parameter_list=model.parameter_list())

place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

for data in loader():
    cost_val = exe.run(fleet.main_program,
                       feed=data,
                       fetch_list=[model.loss.name])
    print(cost_val[0])
