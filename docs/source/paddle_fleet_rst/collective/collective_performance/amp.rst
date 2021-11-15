单精度浮点数计算
==========================

简介
----

传统上，深度学习训练通常使用32比特双精度浮点数\ ``FP32`` \ 作为参数、梯度和中间Activation等的数据存储格式。使用\ ``FP32``\ 作为数据存储格式，每个数据需要4个字节的存储空间。为了节约显存消耗，业界提出使用16比特单精度浮点数\ ``FP16``\ 作为数据存储格式。使用\ ``FP16``\ 作为数据存储格式，每个数据仅需要2个字节的存储空间，相比于\ ``FP32``\ 可以节省一般的存储空间。除了降低显存消耗，\ ``FP16``\ 格式下，计算速度通常也更快，因此可以加速训练。

单精度浮点训练可以带来以下好处：

1. 减少对GPU显存的需求，或者在GPU显存保持不变的情况下，可以支持更大模型和更大的batch size；
2. 降低显存读写的带宽压力；
3. 加速GPU数学运算速度 (需要GPU支持\ `[1] <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensorop>`__)；
4. 按照NVIDA数据，GPU上\ ``FP16``\ 计算吞吐量是\ ``FP32``\ 的2~8倍\ `[2] <https://arxiv.org/abs/1710.03740>`__\ 。

自动混合精度原理
----

飞桨中，我们引入自动混合精度(Auto Mixed Precision, AMP)，混合使用\ ``FP32``\ 和\ ``FP16``\ ，在保持训练精度的同时，进一步提升训练的速度。实现了 ``自动维护FP32 、FP16参数副本``,
``Dynamic loss scaling``, ``op黑白名单`` 等策略来避免
因 FP16 动态范围较小而带来的模型最终精度损失。 Fleet 作为Paddle通用的分布式训练API提供了简单易用的接口, 用户只需要添加几行代码
就可将自动混合精度应用到原有的分布式训练中进一步提升训练速度.

首先介绍半精度（FP16）。如下图所示，半精度（FP16）是一种相对较新的浮点类型，在计算机中使用2字节（16位）存储。
在IEEE 754-2008标准中，它亦被称作binary16。与计算中常用的单精度（FP32）和双精度（FP64）类型相比，FP16更适于在精度要求不高的场景中使用。

.. image:: ../img/amp.png
  :width: 400
  :alt: amp
  :align: center

在使用相同的超参数下，混合精度训练使用半精度浮点（FP16）和单精度（FP32）浮点即可达到与使用纯单精度训练相同的准确率，并可加速模型的训练速度。这主要得益于英伟达推出的Volta及Turing架构GPU在使用FP16计算时具有如下特点：

- FP16可降低一半的内存带宽和存储需求，这使得在相同的硬件条件下研究人员可使用更大更复杂的模型以及更大的batch size大小。

- FP16可以充分利用英伟达Volta及Turing架构GPU提供的Tensor Cores技术。在相同的GPU硬件上，Tensor Cores的FP16计算吞吐量是FP32的8倍。

静态图操作实践
----

为了使用AMP，只需要打开相应的配置选项：

.. code:: python

    strategy = fleet.DistributedStrategy()
    strategy.amp = True
    strategy.amp_configs = {
        "init_loss_scaling": 32768,
        "decr_every_n_nan_or_inf": 2,
        "incr_every_n_steps": 1000,
        "incr_ratio": 2.0,
        "use_dynamic_loss_scaling": True,
        "decr_ratio": 0.5,
        "custom_white_list": [],
        "custom_black_list": [],
    }

上述例子存放在：`example/resnet/train_fleet_static_amp.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_static_amp.py>`_。
假设要运行8卡的任务，那么只需在命令行中执行:

.. code-block:: sh

   python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 train_fleet_static_amp.py

您将看到显示如下日志信息：

.. code-block::

   -----------  Configuration Arguments -----------
   gpus: None
   heter_worker_num: None
   heter_workers:
   http_port: None
   ips: 127.0.0.1
   log_dir: log
   ...
   ------------------------------------------------
   ...
   INFO 2021-01-19 14:46:03,186 launch_utils.py:472] Local start 8 processes. First process distributed environment info (Only For Debug):
      +=======================================================================================+
      |                        Distributed Envs                      Value                    |
      +---------------------------------------------------------------------------------------+
      |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:54114               |
      |                     PADDLE_TRAINERS_NUM                        2                      |
      |                PADDLE_TRAINER_ENDPOINTS  ... 0.1:24697,127.0.0.1:53564,127.0.0.1:37181|
      |                     FLAGS_selected_gpus                        0                      |
      |                       PADDLE_TRAINER_ID                        0                      |
      +=======================================================================================+
   W0119 14:46:16.315114 84038 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 9.2
   W0119 14:46:16.320163 84038 device_context.cc:372] device: 0, cuDNN Version: 7.4.
   W0119 14:46:25.249166 84038 fuse_all_reduce_op_pass.cc:79] Find all_reduce operators: 161. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 8.
   [Epoch 0, batch 0] loss: 0.19354, acc1: 0.00000, acc5: 0.00000
   [Epoch 0, batch 5] loss: 0.20044, acc1: 0.00000, acc5: 0.00000

需要注意的是，不同飞桨版本，上述信息可能会有所差异。

动态图操作实践
----

使用飞桨框架提供的API，paddle.amp.auto_cast 和 paddle.amp.GradScaler 能够实现自动混合精度训练（Automatic Mixed Precision，AMP），
即在相关OP的计算中，自动选择FP16或FP32计算。开启AMP模式后，使用FP16与FP32进行计算的OP列表可见该 `[3] <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/Overview_cn.html>`__。

下面来看一个具体的例子，来了解如果使用飞桨框架实现混合精度训练。

首先定义辅助函数，用来计算训练时间。

.. code-block:: python

   import time

   # 开始时间
   start_time = None

   def start_timer():
      # 获取开始时间
      global start_time
      start_time = time.time()

   def end_timer_and_print(msg):
      # 打印信息并输出训练时间
      end_time = time.time()
      print("\n" + msg)
      print("共计耗时 = {:.3f} sec".format(end_time - start_time))

构建一个简单的网络，用于对比使用普通方法进行训练与使用混合精度训练的训练速度。该网络由三层 Linear 组成，其中前两层 Linear 后接 ReLU 激活函数。

.. code-block:: python

   import paddle
   import paddle.nn as nn

   class SimpleNet(nn.Layer):

      def __init__(self, input_size, output_size):
         super(SimpleNet, self).__init__()
         self.linear1 = nn.Linear(input_size, output_size)
         self.relu1 = nn.ReLU()
         self.linear2 = nn.Linear(input_size, output_size)
         self.relu2 = nn.ReLU()
         self.linear3 = nn.Linear(input_size, output_size)

      def forward(self, x):

         x = self.linear1(x)
         x = self.relu1(x)
         x = self.linear2(x)
         x = self.relu2(x)
         x = self.linear3(x)

         return x

设置训练的相关参数，这里为了能有效的看出混合精度训练对于训练速度的提升，将 input_size 与 output_size 的值设为较大的值，为了使用GPU 提供的Tensor Core 性能，还需将 batch_size 设置为 8 的倍数。

.. code-block:: python

   epochs = 5
   input_size = 4096   # 设为较大的值
   output_size = 4096  # 设为较大的值
   batch_size = 512    # batch_size 为8的倍数
   nums_batch = 50

   train_data = [paddle.randn((batch_size, input_size)) for _ in range(nums_batch)]
   labels = [paddle.randn((batch_size, output_size)) for _ in range(nums_batch)]

   mse = paddle.nn.MSELoss()

使用默认的训练方式进行训练

.. code-block:: python

   model = SimpleNet(input_size, output_size)  # 定义模型

   optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

   start_timer() # 获取训练开始时间

   for epoch in range(epochs):
      datas = zip(train_data, labels)
      for i, (data, label) in enumerate(datas):

         output = model(data)
         loss = mse(output, label)

         # 反向传播
         loss.backward()

         # 训练模型
         optimizer.step()
         optimizer.clear_grad()

   print(loss)
   end_timer_and_print("默认耗时:") # 获取结束时间并打印相关信息

.. code-block:: bash

   Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
       [1.25010288])

   默认耗时:
   共计耗时 = 2.943 sec

使用AMP训练模型

在飞桨框架中，使用自动混合精度训练，需要进行三个步骤：

- Step1： 定义 GradScaler ，用于缩放 loss 比例，避免浮点数下溢

- Step2： 使用 auto_cast 用于创建AMP上下文环境，该上下文中自动会确定每个OP的输入数据类型（FP16或FP32）

- Step3： 使用 Step1中定义的 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播，完成训练

.. code-block:: python

   model = SimpleNet(input_size, output_size)  # 定义模型

   optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

   # Step1：定义 GradScaler，用于缩放loss比例，避免浮点数溢出
   scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

   start_timer() # 获取训练开始时间

   for epoch in range(epochs):
      datas = zip(train_data, labels)
      for i, (data, label) in enumerate(datas):

         # Step2：创建AMP上下文环境，开启自动混合精度训练
         with paddle.amp.auto_cast():
               output = model(data)
               loss = mse(output, label)

         # Step3：使用 Step1中定义的 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
         scaled = scaler.scale(loss)
         scaled.backward()

         # 训练模型
         scaler.minimize(optimizer, scaled)
         optimizer.clear_grad()

   print(loss)
   end_timer_and_print("使用AMP模式耗时:")

.. code-block:: bash

   Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
       [1.23644269])

   使用AMP模式耗时:
   共计耗时 = 1.222 sec

上述例子存放在：`example/amp/amp_dygraph.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/amp/amp_dygraph.py>`_。
