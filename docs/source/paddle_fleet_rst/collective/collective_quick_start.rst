
快速开始
--------

Collective训练快速开始
^^^^^^^^^^^^^^^^^^^^^^

本节将以CV领域经典模型ResNet50为例，介绍如何使用Fleet API（paddle.distributed.fleet）完成Collective分布式训练。我们采用Paddle内置的flowers数据集和Momentum优化器方法，循环迭代10个epoch，并在每个step打印当前模型的损失值和精度值。具体代码请参考\ `FleetX/examples/resnet <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet>`_\ ，其中包含动态图和静态图两种执行方式。resnet_dygraph.py为动态图模型相关代码，train_fleet_dygraph.py为动态图训练脚本。resnet_static.py为静态图模型相关代码，train_fleet_static.py为静态图训练脚本。

版本要求
^^^^^^^^

在编写分布式训练程序之前，用户需要确保已经安装paddlepaddle-2.0.0-cpu或paddlepaddle-2.0.0-gpu及以上版本的飞桨开源框架。关于如何安装paddlepaddle框架，请参考\ `安装指南 <https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html>`_\ 。

操作方法
^^^^^^^^

与单机单卡的普通模型训练相比，无论静态图还是动态图，Collective训练的代码都只需要补充三个部分代码：

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


动态图完整代码
~~~~~~~~

train_fleet_dygraph.py的完整训练代码如下所示。

.. code-block:: py

    # -*- coding: UTF-8 -*-
    import numpy as np
    import paddle
    # 导入必要分布式训练的依赖包
    from paddle.distributed import fleet
    # 导入模型文件
    from paddle.vision.models import ResNet
    from paddle.vision.models.resnet import BottleneckBlock
    from paddle.io import Dataset, BatchSampler, DataLoader

    base_lr = 0.1   # 学习率
    momentum_rate = 0.9 # 冲量
    l2_decay = 1e-4 # 权重衰减

    epoch = 10  #训练迭代次数
    batch_num = 100 #每次迭代的batch数
    batch_size = 32 #训练批次大小
    class_dim = 102

    # 设置数据读取器
    class RandomDataset(Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __getitem__(self, idx):
            image = np.random.random([3, 224, 224]).astype('float32')
            label = np.random.randint(0, class_dim - 1, (1, )).astype('int64')
            return image, label

        def __len__(self):
            return self.num_samples

    # 设置优化器
    def optimizer_setting(parameter_list=None):
        optimizer = paddle.optimizer.Momentum(
            learning_rate=base_lr,
            momentum=momentum_rate,
            weight_decay=paddle.regularizer.L2Decay(l2_decay),
            parameters=parameter_list)
        return optimizer

    # 设置训练函数
    def train_resnet():
        # 初始化Fleet环境
        fleet.init(is_collective=True)

        resnet = ResNet(BottleneckBlock, 50, num_classes=class_dim)

        optimizer = optimizer_setting(parameter_list=resnet.parameters())
        optimizer = fleet.distributed_optimizer(optimizer)
        # 通过Fleet API获取分布式model，用于支持分布式训练
        resnet = fleet.distributed_model(resnet)

        dataset = RandomDataset(batch_num * batch_size)
        train_loader = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=2)

        for eop in range(epoch):
            resnet.train()
            
            for batch_id, data in enumerate(train_loader()):
                img, label = data
                label.stop_gradient = True

                out = resnet(img)
                loss = paddle.nn.functional.cross_entropy(input=out, label=label)
                avg_loss = paddle.mean(x=loss)
                acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
                acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
                
                avg_loss.backward()
                optimizer.step()
                resnet.clear_gradients()

                if batch_id % 5 == 0:
                    print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, avg_loss, acc_top1, acc_top5))
    # 启动训练
    if __name__ == '__main__':
        train_resnet()


静态图完整代码
~~~~~~~~

train_fleet_static.py的完整训练代码如下所示。

.. code-block:: py

   # -*- coding: UTF-8 -*-
   import numpy as np
   import paddle
   # 导入必要分布式训练的依赖包
   import paddle.distributed.fleet as fleet
   # 导入模型文件
   from paddle.vision.models import ResNet
   from paddle.vision.models.resnet import BottleneckBlock
   from paddle.io import Dataset, BatchSampler, DataLoader
   import os

   base_lr = 0.1   # 学习率
   momentum_rate = 0.9 # 冲量
   l2_decay = 1e-4 # 权重衰减

   epoch = 10  #训练迭代次数
   batch_num = 100 #每次迭代的batch数
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
   class RandomDataset(Dataset):
       def __init__(self, num_samples):
           self.num_samples = num_samples

       def __getitem__(self, idx):
           image = np.random.random([3, 224, 224]).astype('float32')
           label = np.random.randint(0, class_dim - 1, (1, )).astype('int64')
           return image, label

       def __len__(self):
           return self.num_samples

   def get_train_loader(place):
       dataset = RandomDataset(batch_num * batch_size)
       train_loader = DataLoader(dataset,
                    places=place,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=2)
       return train_loader
   
   # 设置训练函数
   def train_resnet():
       paddle.enable_static() # 使能静态图功能
       paddle.vision.set_image_backend('cv2')

       image = paddle.static.data(name="x", shape=[None, 3, 224, 224], dtype='float32')
       label= paddle.static.data(name="y", shape=[None, 1], dtype='int64')
       # 调用ResNet50模型
       model = ResNet(BottleneckBlock, 50, num_classes=class_dim)
       out = model(image)
       avg_cost = paddle.nn.functional.cross_entropy(input=out, label=label)
       acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
       acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
       # 设置训练资源，本例使用GPU资源
       place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))

       train_loader = get_train_loader(place)
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
           for batch_id, (image, label) in enumerate(train_loader()):
               loss, acc1, acc5 = exe.run(paddle.static.default_main_program(), feed={'x': image, 'y': label}, fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])             
               if batch_id % 5 == 0:
                   print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, loss, acc1, acc5))
   # 启动训练
   if __name__ == '__main__':
       train_resnet()

当使用\ ``paddle.distributed.launch``\ 组件启动飞桨分布式任务时，在静态图模式下，可以通过\ ``FLAGS_selected_gpus``\ 环境变量获取当前进程绑定的GPU卡，如上面的例子所示。

运行示例
^^^^^^^^

通过\ ``paddle.distributed.launch``\ 组件启动飞桨分布式任务，假设要运行2卡的任务，那么只需在命令行中执行:

动态图：

.. code-block::

   python -m paddle.distributed.launch --gpus=0,1 train_fleet_dygraph.py

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
    launch train in GPU mode
    INFO 2021-03-23 14:11:38,107 launch_utils.py:481] Local start 2 processes. First process distributed environment info (Only For Debug):
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:59648               |
        |                     PADDLE_TRAINERS_NUM                        2                      |
        |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:59648,127.0.0.1:50871       |
        |                     FLAGS_selected_gpus                        0                      |
        |                       PADDLE_TRAINER_ID                        0                      |
        +=======================================================================================+

    I0323 14:11:39.383992  3788 nccl_context.cc:66] init nccl context nranks: 2 local rank: 0 gpu id: 0 ring id: 0
    W0323 14:11:39.872674  3788 device_context.cc:368] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 9.2
    W0323 14:11:39.877283  3788 device_context.cc:386] device: 0, cuDNN Version: 7.4.
    [Epoch 0, batch 0] loss: 4.77086, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 5] loss: 15.69098, acc1: 0.03125, acc5: 0.18750
    [Epoch 0, batch 10] loss: 23.41379, acc1: 0.00000, acc5: 0.09375
    ...

静态图：

.. code-block::

   python -m paddle.distributed.launch --gpus=0,1 train_fleet_static.py

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

请注意，不同飞桨版本上述显示信息可能会略有不同。

单机八卡训练启动命令类似，只需正确指定\ ``gpus``\ 参数即可，如下所示：

.. code-block::
   
   # 动态图
   python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 train_fleet_dygraph.py
   
   # 静态图
   python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 train_fleet_static.py


从单机多卡到多机多卡训练，在代码上不需要做任何改动，只需再额外指定\ ``ips``\ 参数即可。其内容为多机的IP列表，命令如下所示（假设两台机器的ip地址分别为192.168.0.1和192.168.0.2）：

.. code-block::

   # 动态图
   python -m paddle.distributed.launch --ips="192.168.0.1,192.168.0.2" --gpus 0,1,2,3,4,5,6,7 train_fleet_dygraph.py

    # 静态图
   python -m paddle.distributed.launch --ips="192.168.0.1,192.168.0.2" --gpus 0,1,2,3,4,5,6,7 train_fleet_static.py

了解更多启动分布式训练任务信息，请参考\ `分布式任务启动方法 <../launch.html>`_\ 。
