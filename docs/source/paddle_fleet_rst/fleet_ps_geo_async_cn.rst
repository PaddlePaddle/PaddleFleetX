低频通信参数服务器训练算法
==========================

简介
----

众所周知，在同步/异步参数服务器分布式训练中Trainer每训练完一个周期，都会将梯度上传至Server，等待Server分发最新的参数后才开始新一轮的训练。在这种训练方式中，节点间的通信会消耗大量的时间成本，进而影响训练的效率。

为了降低节点见通信对训练速度的影响，Fleet提供了一种更高效的参数更新策略：GeoSGD

原理
----

在GeoSGD更新策略中，Trainer的参数更新也是在全异步的条件下进行的。但与异步参数服务器有以下不同：

-  与普通的参数服务器不同，在GEO策略中，每个Trainer负责在本地维护自己的参数更新，在训练一定数量的步数后将本轮训练出的参数与上一轮结束后的参数做差。并除以Trainer的个数，将结果上传至Server。Server则负责为每个Trainer计算其参数与全局参数的diff。

-  GEO更新策略会在训练过程中启动多个进程，负责参数更新及节点通信。在Trainer与Server的整个交互过程中，主进程会保持模型的训练，由子进程负责与server进行交互，在拿到与全局参数的diff后将其更新至主进程。

GEO策略通过模型训练与节点通信同步进行的方式，在保证模型效果的前提下，大大提升了训练的速度。经过在SGD优化器上的测试，GEO策略相比异步参数服务器，训练速度提高了1倍。

接下来我们将通过例子为您讲解GEO在Fleet中是如何应用的。

在开始之前我们首先需要下载训练中所需要的数据：

.. code:: sh

   # 下载并解压数据，训练数据讲保存至名为 raw_data 的文件夹
   wget --no-check-certificate https://fleet.bj.bcebos.com/ctr_data.tar.gz
   tar -zxvf ctr_data.tar.gz

操作实践
--------

添加依赖
~~~~~~~~

首先我们需要添加训练中所用到的python模块，\ ``fleetx``
可以用于加载我们为用户封装的接口如：加载模型及数据，模型训练等。\ ``paddle.distributed.fleet``
中定义了丰富的分布式策略供用户使用。

.. code:: python

   import fleetx as X
   import paddle.fluid as fluid
   import paddle.distributed.fleet as fleet
   import paddle.distributed.fleet.base.role_maker as role_maker

定义分布式模式并初始化
~~~~~~~~~~~~~~~~~~~~~~

通过\ ``X.parse_train_configs()``\ 接口，用户可以定义训练相关的参数，如：学习率、衰减率等。同时通过\ ``fleet.init()``\ 接口定义了分布式模式，定义GEO策略使用的初始化接口与同步/异步参数服务器相同，都是\ ``init()``\ 默认的模式。

.. code:: python

   configs = X.parse_train_configs()
   role = role_maker.PaddleCloudRoleMaker()
   fleet.init(role)

加载模型及数据
~~~~~~~~~~~~~~

在这个例子中我们使用了与同步/异步参数服务器相同的CTR-DNN模型。用\ ``X.applications``\ 接口加载模型，并加载定制化的数据。

.. code:: python

   model = X.applications.MultiSlotCTR()
   loader = model.load_multislot_from_file('./train_data')

定义同步训练 Strategy 及 Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在Fleet
API中，用户可以使用\ ``fleet.DistributedStrategy()``\ 接口定义自己想要使用的分布式策略。

想要使用GEO策略，用户首先需要打开异步参数服务器开关，即设置\ ``a_sync``\ 为
True。

然后用户需要通过\ ``dist_strategy.a_sync_configs``\ 设置Trainer上传参数的频率，下面的代码中我们设置Trainer每训练10000个Batch后与Server进行交互。

.. code:: python

   dist_strategy = fleet.DistributedStrategy()
   dist_strategy.a_sync = True
   dist_strategy.a_sync_configs = {"k_steps": 10000}

   optimizer = fluid.optimizer.SGD(learning_rate=0.0001)

   optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
   optimizer.minimize(model.loss)

开始训练
~~~~~~~~

GEO策略的训练代码沿用了参数服务器分布式训练的形式。

对于Server节点，首先用\ ``init_server()``\ 接口对其进行初始化，然后启动服务并开始监听由训练节点传来的参数变化值。

同样对于训练节点，用\ ``init_worker()``\ 接口进行初始化后x，开始执行训练任务。运行\ ``X.Trainer.fit``\ 接口开始训练。

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
