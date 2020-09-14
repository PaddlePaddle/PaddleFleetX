使用Fleet进行参数服务器训练
===========================

在大数据浪潮的推动下，有标签训练数据的规模取得了飞速的增长。现在人们通常用数百万甚至上千万的有标签图像来训练图像分类器（如，ImageNet包含1400万幅图像，涵盖两万多个种类），用成千上万小时的语音数据来训练语音模型（如，Deep
Speech
2系统使用了11940小时的语音数据以及超过200万句表述来训练语音识别模型）。在真实的业务场景中，训练数据的规模可以达到上述数据集的数十倍甚至数百倍，如此庞大的数据需要消耗大量的计算资源和训练时间使模型达到收敛状态（数天时间）。

为了提高模型的训练效率，分布式训练应运而生，其中基于参数服务器的分布式训练为一种常见的中心化共享参数的同步方式。与单机训练不同的是在参数服务器分布式训练中，各个节点充当着不同的角色：

-  **训练节点**\ ：该节点负责完成数据读取、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。
-  **服务节点**\ ：在收到所有训练节点传来的梯度后，该节点会将梯度聚合并更新参数。最后将参数发送给训练节点，开始新一轮的训练。

根据参数更新的方式不同，可以分为同步和异步两种：

-  **同步训练**\ ：在同步参数服务器分布式训练中，所有训练节点的进度保持一致。每训练完一个Batch后，训练节点会上传梯度，然后开始等待更新后的参数。服务节点拿到所有训练节点上传的梯度后，才会对参数进行更新。因此，在任何一个时间点，所有训练节点都处于相同的训练阶段。
-  **异步训练**\ ：与同步训练不同，在异步训练中任何两个训练节点之间的参数更新都互不影响。每一个训练节点完成训练、上传梯度后，服务节点都会立即更新参数并将结果返回至相应的训练节点。拿到最新的参数后，该训练节点会立即开始新一轮的训练。

下面我们将通过例子，为您介绍同步/异步训练在Fleet中的实现。

在开始之前我们首先需要下载训练中所需要的数据：

.. code:: sh

   # 下载并解压数据，训练数据讲保存至名为 raw_data 的文件夹
   wget --no-check-certificate https://fleet.bj.bcebos.com/ctr_data.tar.gz
   tar -zxvf ctr_data.tar.gz

实用样例
--------

下面我们来介绍如何用Fleet接口，完成参数服务器分布式训练（假设训练脚本为ctr_app.py）。

导入依赖
~~~~~~~~

.. code:: python

   import os
   import fleetx as X
   import paddle.fluid as fluid
   import paddle.distributed.fleet as fleet
   import paddle.distributed.fleet.base.role_maker as role_maker

定义分布式模式并初始化
~~~~~~~~~~~~~~~~~~~~~~

通过\ ``X.parse_train_configs()``\ 接口，用户可以定义训练相关的参数，如：学习率、衰减率等。同时通过\ ``fleet.init()``\ 接口定义了分布式模型，\ ``init()``\ 接口默认使用参数服务器模式，所以用户不需要定
义任何参数。

.. code:: python

   configs = X.parse_train_configs()
   role = role_maker.PaddleCloudRoleMaker()
   fleet.init(role)

加载模型及数据
~~~~~~~~~~~~~~

用户可以通过\ ``X.applications``\ 接口加载我们预先定义好的模型。在这个例子中我们将使用CTR-DNN模型，同时用户可以为模型定制的data_loader接口加载数据.

.. code:: python

   model = X.applications.MultiSlotCTR()
   loader = model.load_multislot_from_file('./train_data')

定义同步训练 Strategy 及 Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在Fleet
API中，用户可以使用\ ``fleet.DistributedStrategy()``\ 接口定义自己想要使用的分布式策略。

其中\ ``a_sync``\ 选项用于定义参数服务器相关的策略，当其被设定为\ ``False``\ 时，分布式训练将在同步的模式下进行。反之，当其被设定成\ ``True``\ 时，分布式训练将在异步的模式下进行。

.. code:: python

   dist_strategy = fleet.DistributedStrategy()
   dist_strategy.a_sync = False

   optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
   optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
   optimizer.minimize(model.loss)

开始训练
~~~~~~~~

完成模型及训练策略以后，我们就可以开始训练模型了。因为在参数服务器模式下会有不同的角色，所以根据不同节点分配不同的任务。

对于服务器节点，首先用\ ``init_server()``\ 接口对其进行初始化，然后启动服务并开始监听由训练节点传来的梯度。

同样对于训练节点，用\ ``init_worker()``\ 接口进行初始化后，
开始执行训练任务。运行\ ``X.Trainer.fit``\ 接口开始训练，并得到训练中每一步的损失值。

.. code:: python

   if fleet.is_server():
       fleet.init_server()
       fleet.run_server()
   else:
       fleet.init_worker()
       trainer = X.Trainer(fluid.CPUPlace())
       trainer.fit(model, loader, epoch=10)

运行训练脚本
~~~~~~~~~~~~

定义完训练脚本后，我们就可以用\ ``fleetrun``\ 指令运行分布式任务了。其中\ ``server_num``,
``worker_num``\ 分别为服务节点和训练节点的数量。在本例中，服务节点有1个，训练节点有两个。

.. code:: sh

   fleetrun --server_num=1 --worker_num=2 ctr_app.py
