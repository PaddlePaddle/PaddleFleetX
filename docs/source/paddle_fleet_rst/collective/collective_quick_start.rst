
快速开始
--------

Collective训练快速开始
^^^^^^^^^^^^^^^^^^^^^^

本节将采用CV领域非常经典的模型ResNet50为例，介绍如何使用Fleet API（paddle.distributed.fleet）完成Collective训练任务。数据方面我们采用Paddle内置的flowers数据集，优化器使用Momentum方法。循环迭代多个epoch，每轮打印当前网络具体的损失值和acc值。具体代码保存在\ `FleetX/examples/resnet <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet>`_\ 下面，其中resnet_static.py用于保存模型相关代码，而train_fleet_static.py为本节需要讲解的训练脚本。

版本要求
^^^^^^^^

在编写分布式训练程序之前，用户需要确保已经安装paddlepaddle-2.0.0-rc-cpu或paddlepaddle-2.0.0-rc-gpu及以上版本的飞桨开源框架。

操作方法
^^^^^^^^

与单机单卡的普通模型训练相比，Collective训练的代码主要需要补充三个部分代码：


#. 导入分布式训练需要的依赖包。
#. 初始化Fleet环境。
#. 设置分布式训练需要的优化器。
   下面将逐一进行讲解。

导入依赖
~~~~~~~~

导入必要的依赖，例如分布式训练专用的Fleet API(paddle.distributed.fleet)。

.. code-block::

   from paddle.distributed import fleet

初始化fleet环境
~~~~~~~~~~~~~~~

包括定义缺省的分布式策略，然后通过将参数is_collective设置为True，使训练架构设定为Collective架构。

.. code-block::

   strategy = fleet.DistributedStrategy()
   fleet.init(is_collective=True, strategy=strategy)

设置分布式训练使用的优化器
~~~~~~~~~~~~~~~~~~~~~~~~~~

使用distributed_optimizer设置分布式训练优化器。

.. code-block::

   optimizer = fleet.distributed_optimizer(optimizer)

完整代码
~~~~~~~~

train_fleet_static.py的完整训练代码如下所示。

.. code-block:: py

   # -*- coding: UTF-8 -*-
   import numpy as np
   import argparse
   import ast
   import paddle
   # 导入必要分布式训练的依赖包
   import paddle.distributed.fleet as fleet
   # 导入模型文件
   import resnet_static as resnet
   import os

   base_lr = 0.1   # 学习率
   momentum_rate = 0.9 # 冲量
   l2_decay = 1e-4 # 权重衰减

   epoch = 10  #训练迭代次数
   batch_size = 32 #训练批次大小
   class_dim = 10

   # 设置优化器
   def optimizer_setting(parameter_list=None):
       optimizer = paddle.optimizer.Momentum(
           learning_rate=base_lr,
           momentum=momentum_rate,
           weight_decay=paddle.regularizer.L2Decay(l2_decay),
           parameters=parameter_list)
       return optimizer
   # 设置数据读取器
   def get_train_loader(feed_list, place):
       def reader_decorator(reader):
           def __reader__():
               for item in reader():
                   img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
                   label = np.array(item[1]).astype('int64').reshape(1)
                   yield img, label

           return __reader__
       train_reader = paddle.batch(
               reader_decorator(paddle.dataset.flowers.train(use_xmap=True)),
               batch_size=batch_size,
               drop_last=True)
       train_loader = paddle.io.DataLoader.from_generator(
           capacity=32,
           use_double_buffer=True,
           feed_list=feed_list,
           iterable=True)
       train_loader.set_sample_list_generator(train_reader, place)
       return train_loader
   # 设置训练函数
   def train_resnet():
       paddle.enable_static() # 使能静态图功能
       paddle.vision.set_image_backend('cv2')

       image = paddle.static.data(name="x", shape=[None, 3, 224, 224], dtype='float32')
       label= paddle.static.data(name="y", shape=[None, 1], dtype='int64')
       # 调用ResNet50模型
       model = resnet.ResNet(layers=50)
       out = model.net(input=image, class_dim=class_dim)
       avg_cost = paddle.nn.functional.cross_entropy(input=out, label=label)
       acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
       acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
       # 设置训练资源，本例使用GPU资源
       place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))

       train_loader = get_train_loader([image, label], place)
       #初始化Fleet环境
       strategy = fleet.DistributedStrategy()
       fleet.init(is_collective=True, strategy=strategy)
       optimizer = optimizer_setting()

       # 通过Fleet API获取分布式优化器，将参数传入飞桨的基础优化器
       optimizer = fleet.distributed_optimizer(optimizer)
       optimizer.minimize(avg_cost)

       exe = paddle.static.Executor(place)
       exe.run(paddle.static.default_startup_program())

       epoch = 10
       step = 0
       for eop in range(epoch):
           for batch_id, data in enumerate(train_loader()):
               loss, acc1, acc5 = exe.run(paddle.static.default_main_program(), feed=data, fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])             
               if batch_id % 5 == 0:
                   print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, loss, acc1, acc5))
   # 启动训练
   if __name__ == '__main__':
       train_resnet()

运行示例
^^^^^^^^

假设要运行2卡的任务，那么只需在命令行中执行:

.. code-block::

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

单机八卡训练启动命令类似，只需正确指定\ ``gpus``\ 参数即可，如下所示：

.. code-block::

   fleetrun --gpus 0,1,2,3,4,5,6,7 train_fleet_static.py

从单机多卡到多机多卡训练，在代码上不需要做任何改动，只需再额外指定ips参数即可。其内容为多机的ip列表，命令如下所示：

.. code-block::

   fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus 0,1,2,3,4,5,6,7 train_fleet_static.py
