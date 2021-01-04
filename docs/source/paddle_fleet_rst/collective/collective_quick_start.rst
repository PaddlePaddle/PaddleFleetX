
静态图分布式训练快速开始
------------------------

对于大部分用户来讲，数据并行训练基本可以解决实际业务中的训练要求。我们以一个非常经典的神经网络为例，介绍如何使用飞桨高级分布式API ``paddle.distributed.fleet``\ 进行数据并行训练。在数据并行方式下，通常可以采用两种架构进行并行训练，即集合通信训练（Collective Training）和参数服务器训练（Parameter Server Training），接下来的例子会分别说明两种架构的数据并行是如何实现的。

版本要求
^^^^^^^^


* paddlepaddle-2.0.0-rc-cpu / paddlepaddle-2.0.0-rc-gpu及以上

模型描述
^^^^^^^^

我们采用CV领域非常经典的模型ResNet50为例进行介绍。数据方面我们采用\ ``Paddle``\ 内置的\ ``flowers``\ 数据集，优化器使用Momentum方法。循环迭代多个epoch，每轮打印当前网络具体的损失值和acc值。代码整体存放在\ `example/resnet <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet>`_\ 下面。

单机单卡训练
^^^^^^^^^^^^

下面是一个单机单卡程序的示例。

.. code-block:: py

   # -*- coding: UTF-8 -*-
   import paddle
   def train_resnet():
      # 1.开启静态图模式
      paddle.enable_static()

      # 2. 定义网络对象，损失函数和优化器
      paddle.vision.set_image_backend('cv2')
      image = paddle.static.data(name="x", shape=[None, 3, 224, 224], dtype='float32')
      label= paddle.static.data(name="y", shape=[None, 1], dtype='int64')
      model = resnet.ResNet(layers=50)
      out = model.net(input=image, class_dim=class_dim)
      avg_cost = paddle.nn.functional.cross_entropy(input=out, label=label)
      acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
      acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

      place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
      train_loader = get_train_loader([image, label], place)

      optimizer = optimizer_setting()
      optimizer.minimize(avg_cost)

      exe = paddle.static.Executor(place)
      exe.run(paddle.static.default_startup_program())

      # 3. 进行模型训练（前向、反向、参数更新），并打印相关结果
      epoch = 10
      step = 0
      for eop in range(epoch):
         for batch_id, data in enumerate(train_loader()):
            loss, acc1, acc5 = exe.run(paddle.static.default_main_program(), feed=data, fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])             
            print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, loss, acc1, acc5))

单机多卡训练
^^^^^^^^^^^^

使用Fleet接口进行动态图分布式训练其实非常的简单，只需修改3个步骤：


#. 
   导入\ ``paddle.distributed.fleet``\ 包

   .. code-block:: py

      from paddle.distributed import fleet

#. 
   初始化fleet环境

   .. code-block:: py

      strategy = fleet.DistributedStrategy()
      fleet.init(is_collective=True, strategy=strategy)

#. 
   通过fleet获取分布式优化器，参数传入paddle的基础优化器

   .. code-block:: py

      optimizer = fleet.distributed_optimizer(optimizer)

根据我们最开始提供的单机单卡代码示例，再根据3步口诀进行修改，完整的单机多卡示例代码如下：

.. code-block:: py

   # -*- coding: UTF-8 -*-
   import paddle
   # 1. 导入`paddle.distributed.fleet`包
   from paddle.distributed import fleet

   def train_resnet():
      paddle.enable_static()
      paddle.vision.set_image_backend('cv2')

      image = paddle.static.data(name="x", shape=[None, 3, 224, 224], dtype='float32')
      label= paddle.static.data(name="y", shape=[None, 1], dtype='int64')

      model = resnet.ResNet(layers=50)
      out = model.net(input=image, class_dim=class_dim)
      avg_cost = paddle.nn.functional.cross_entropy(input=out, label=label)
      acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
      acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

      place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))

      train_loader = get_train_loader([image, label], place)

      # 2. 初始化fleet环境
      strategy = fleet.DistributedStrategy()
      fleet.init(is_collective=True, strategy=strategy)
      optimizer = optimizer_setting()

      # 3. 通过fleet获取分布式优化器，参数传入paddle的基础优化器
      optimizer = fleet.distributed_optimizer(optimizer)
      optimizer.minimize(avg_cost)

      exe = paddle.static.Executor(place)
      exe.run(paddle.static.default_startup_program())

      epoch = 10
      step = 0
      for eop in range(epoch):
         for batch_id, data in enumerate(train_loader()):
            loss, acc1, acc5 = exe.run(paddle.static.default_main_program(), feed=data, fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])             
            print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, loss, acc1, acc5))

上述例子的完整代码存放在：\ `train_fleet_static.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_static.py>`_\ 下面。假设要运行2卡的任务，那么只需在命令行中执行:

.. code-block:: sh

   fleetrun --gpus=0,1 train_fleet_static.py

您将看到显示如下日志信息：

.. code-block::

   -----------  Configuration Arguments -----------
   gpus: 0,1
   heter_worker_num: None
   heter_workers:
   http_port: None
   ips: 127.0.0.1
   log_dir: log
   ...
   ------------------------------------------------
   WARNING 2021-01-04 17:59:08,725 launch.py:314] Not found distinct arguments and compiled with cuda. Default use collective mode
   launch train in GPU mode
   INFO 2021-01-04 17:59:08,727 launch_utils.py:472] Local start 2 processes. First process distributed environment info (Only For Debug):
       +=======================================================================================+
       |                        Distributed Envs                      Value                    |
       +---------------------------------------------------------------------------------------+
       |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:17901               |
       |                     PADDLE_TRAINERS_NUM                        2                      |
       |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:17901,127.0.0.1:18846       |
       |                     FLAGS_selected_gpus                        0                      |
       |                       PADDLE_TRAINER_ID                        0                      |
       +=======================================================================================+

   ...
   W0104 17:59:19.018365 43338 device_context.cc:342] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 9.2
   W0104 17:59:19.022523 43338 device_context.cc:352] device: 0, cuDNN Version: 7.4.
   W0104 17:59:23.193490 43338 fuse_all_reduce_op_pass.cc:78] Find all_reduce operators: 161. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 5.
   [Epoch 0, batch 0] loss: 0.12432, acc1: 0.00000, acc5: 0.06250
   [Epoch 0, batch 5] loss: 1.01921, acc1: 0.00000, acc5: 0.00000
   ...

完整2卡的日志信息也可在\ ``./log/``\ 目录下查看。了解更多\ ``fleetrun``\ 的用法可参考左侧文档\ ``fleetrun 启动分布式任务``\ 。


* 单机八卡训练启动命令
  .. code-block:: shell

     fleetrun --gpus 0,1,2,3,4,5,6,7 train_fleet_static.py
