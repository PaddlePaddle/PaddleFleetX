低频通信参数服务器训练算法
==========================

简介
----

众所周知，在同步/异步参数服务器分布式训练中Worker每训练完一个周期，都会将梯度上传至PServer，等待PServer分发最新的参数后才开始新一轮的训练。在这种训练方式中，节点间的通信会消耗大量的时间成本，进而影响训练的效率。

为了降低节点见通信对训练速度的影响，Fleet提供了一种更高效的参数更新策略：GeoSGD

原理
----

.. image:: ../../../_images/ps/geosgd.png
  :width: 600
  :alt: geosgd
  :align: center

在GeoSGD更新策略中，Worker的参数更新也是在全异步的条件下进行的。但与异步参数服务器有以下不同：

-  与普通的参数服务器不同，在GEO策略中，每个Worker负责在本地维护自己的参数更新，在训练一定数量的步数后将本轮训练出的参数与上一轮结束后的参数做差。并除以Worker的个数，将结果上传至PServer。PServer则负责为每个Worker计算其参数与全局参数的diff。

-  GEO更新策略会在训练过程中启动多个进程，负责参数更新及节点通信。在Worker与PServer的整个交互过程中，主进程会保持模型的训练，由子进程负责与PServer进行交互，在拿到与全局参数的diff后将其更新至主进程。

GEO策略通过模型训练与节点通信同步进行的方式，在保证模型效果的前提下，大大提升了训练的速度。在Word2Vec模型上测试发现，GEO策略相比异步参数服务器，训练速度提高了3倍多。

使用方法
--------

添加依赖
~~~~~~~~

首先我们需要添加训练中所用到的python模块\ ``paddle``\ 和\ ``paddle.distributed.fleet``\ ，后者主要提供分布式相关的接口和策略配置。

目前Paddle默认为动态图运行模式，分布式参数服务器训练当前仅支持在静态图模式下运行，所以需要自行打开静态图开关。

.. code:: python

   import paddle
   import paddle.distributed.fleet as fleet
   paddle.enable_static()

定义模型组网
~~~~~~~~~~~~~~

在这个例子中我们使用Wide&Deep模型。

.. code:: python

    model = WideDeepModel()
    model.net(is_train=True)

初始化分布式训练环境
~~~~~~~~~~~~~~~~~~~~~~

多机参数服务器均是通过\ ``fleet.init()``\ 接口初始化分布式训练环境，用户可通过传入 `role_maker` 进行相关配置，若为None，则框架会自动根据用户在环境变量中的配置进行分布式训练环境的初始化。

.. code:: python

   fleet.init(role_maker=None)
   

配置GEO策略及优化算法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在Fleet API中，用户可以使用\ ``fleet.DistributedStrategy()``\ 接口定义自己想要使用的分布式策略。

想要使用GEO策略，用户首先需要打开异步参数服务器开关，即设置\ ``a_sync``\ 为True。

然后用户需要通过\ ``dist_strategy.a_sync_configs``\ 设置Worker上传参数的频率，下面的代码中我们设置Worker每训练400个Batch后与PServer进行交互。

.. code:: python

   dist_strategy = fleet.DistributedStrategy()
   dist_strategy.a_sync = True
   dist_strategy.a_sync_configs = {"k_steps": 400}

   optimizer = paddle.optimizer.SGD(learning_rate=0.0001)

   optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
   optimizer.minimize(model.cost)

开始训练
~~~~~~~~

GEO策略的训练代码沿用了参数服务器分布式训练的形式。

对于PServer节点，首先用\ ``init_server()``\ 接口对其进行初始化，然后启动服务并开始监听由训练节点传来的参数变化值。

同样对于训练节点，用\ ``init_worker()``\ 接口进行初始化后，开始执行训练任务。

.. code:: python

    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()
    else:
        exe.run(paddle.static.default_startup_program())
        fleet.init_worker()

        # do training
        distributed_training(exe, model)

运行方法
~~~~~~~~~~~~

完整运行示例见 `examples/wide_and_deep`, 需注意，该示例指定的分布式训练模式为异步，可参考GEO模式策略配置方法，将任务运行模式变为GEO模式。

配置完成后，通过\ ``fleetrun``\ 指令运行分布式任务。命令示例如下，其中\ ``server_num``, ``worker_num``\ 分别为服务节点和训练节点的数量。

.. code:: sh

   fleetrun --server_num=2 --worker_num=2 train.py
