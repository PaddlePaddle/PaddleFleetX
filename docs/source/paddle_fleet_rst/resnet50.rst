A Distributed Resnet50 Training Example
---------------------------------------

.. code:: python

   import fleet_lightning as lightning
   import paddle.fluid as fluid
   from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
   import paddle.fluid.incubate.fleet.base.role_maker as role_maker

   configs = lightning.parse_train_configs()

   role = role_maker.PaddleCloudRoleMaker(is_collective=True)
   fleet.init(role)

   model = lightning.applications.Resnet50()

   loader = model.load_imagenet_from_file("/pathto/imagenet/train.txt")

   optimizer = fluid.optimizer.Momentum(learning_rate=configs.lr, momentum=configs.momentum)
   optimizer = fleet.distributed_optimizer(optimizer)
   optimizer.minimize(model.loss)

   place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
   exe = fluid.Executor(place)
   exe.run(fluid.default_startup_program())

   epoch = 30
   for i in range(epoch):
       for data in loader():
               cost_val = exe.run(fleet.main_program, feed=data, fetch_list=[model.loss.name])
