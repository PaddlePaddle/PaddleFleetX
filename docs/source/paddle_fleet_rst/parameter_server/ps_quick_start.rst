快速开始
===========================

在大数据浪潮的推动下，有标签训练数据的规模取得了飞速的增长。现在人们通常用数百万甚至上千万的有标签图像来训练图像分类器（如，ImageNet包含1400万幅图像，涵盖两万多个种类），用成千上万小时的语音数据来训练语音模型（如，Deep
Speech
2系统使用了11940小时的语音数据以及超过200万句表述来训练语音识别模型）。在真实的业务场景中，训练数据的规模可以达到上述数据集的数十倍甚至数百倍，如此庞大的数据需要消耗大量的计算资源和训练时间使模型达到收敛状态（数天时间）。

为了提高模型的训练效率，分布式训练应运而生，其中基于参数服务器的分布式训练为一种常见的中心化共享参数的同步方式。与单机训练不同的是在参数服务器分布式训练中，各个节点充当着不同的角色：

-  **训练节点**\ ：该节点负责完成数据读取、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。
-  **服务节点**\ ：在收到所有训练节点传来的梯度后，该节点会将梯度聚合并更新参数。最后将参数发送给训练节点，开始新一轮的训练。

根据参数更新的方式不同，可以分为同步/异步/Geo异步三种：

-  **同步训练**\ ：所有Worker的进度保持一致，即每训练完一个Batch后，所有Worker会上传梯度给Server，然后开始等待Server返回更新后的参数。Server在拿到所有Worker上传的梯度后，才会开始计算更新后的参数。因此在任何一个时间点，所有Worker都处于相同的训练阶段。同步训练的优势在于Loss可以比较稳定的下降，缺点是整个训练速度较慢，这是典型的木桶原理，速度的快慢取决于最慢的那个Worker的训练计算时间，因此在训练较为复杂的模型时，即模型训练过程中神经网络训练耗时远大于节点间通信耗时的场景下，推荐使用同步训练模式。
-  **异步训练**\ ：与同步训练不同，在异步训练中任何两个Worker之间的参数更新都互不影响。每一个Worker完成训练、上传梯度后，Server都会立即更新参数并将结果返回至相应的训练节点。拿到最新的参数后，该训练节点会立即开始新一轮的训练。异步训练去除了训练过程中的等待机制，训练速度得到了极大的提升，但是缺点也很明显，那就是Loss下降不稳定，容易发生抖动。建议在个性化推荐（召回、排序）、语义匹配等数据量大的场景使用。
-  **GEO异步训练**\ ：GEO异步训练是飞桨独有的一种异步训练模式，训练过程中任何两个训练节点之间的参数更新同样都互不影响，但是每个训练节点本地都会拥有完整的训练流程，即前向计算、反向计算和参数优化，而且每训练到一定的批次(Batch) 训练节点都会将本地的参数计算一次差值(Step间隔带来的参数差值)，将差值发送给服务端累计更新，并拿到最新的参数后，该训练节点会立即开始新一轮的训练。所以显而易见，在GEO异步训练模式下，Worker不用再等待Server发来新的参数即可执行训练，在训练效果和训练速度上有了极大的提升。但是此模式比较适合可以在单机内能完整保存的模型，在搜索、NLP等类型的业务上应用广泛，比较推荐在词向量、语义匹配等场景中使用。

本节将采用推荐领域非常经典的模型wide_and_deep为例，介绍如何使用Fleet API（paddle.distributed.fleet）完成参数服务器训练任务，本次快速开始的示例代码位于https://github.com/PaddlePaddle/FleetX/tree/develop/examples/wide_and_deep。


版本要求
--------
在编写分布式训练程序之前，用户需要确保已经安装paddlepaddle-2.0.0-cpu或paddlepaddle-2.0.0-gpu及以上版本的飞桨开源框架。


操作方法
--------
参数服务器训练的基本代码主要包括如下几个部分：
1. 导入分布式训练需要的依赖包。
2. 定义分布式模式并初始化分布式训练环境。
3. 加载模型及数据。
4. 定义参数更新策略及优化器。
5. 开始训练。

下面将逐一进行讲解。



导入依赖
~~~~~~~~

.. code:: python

    import paddle
    import os
    import paddle.distributed.fleet as fleet
    import paddle.distributed.fleet.base.role_maker as role_maker


定义分布式模式并初始化分布式训练环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


通过\ ``fleet.init()``\ 接口，用户可以定义训练相关的环境，注意此环境是用户预先在环境变量中配置好的，包括：训练节点个数，服务节点个数，当前节点的序号，服务节点完整的IP:PORT列表等。

.. code:: python

    # 当前参数服务器模式只支持静态图模式， 因此训练前必须指定`paddle.enable_static()`
    paddle.enable_static()
    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)


加载模型及数据
~~~~~~~~~~~~~~

.. code:: python

    # 模型定义参考examples/wide_and_deep中model.py
    from model import net
    from reader import data_reader

    feeds, predict, avg_cost = net()

    train_reader = paddle.batch(data_reader(), batch_size=4)
    reader.decorate_sample_list_generator(train_reader)


定义同步训练 Strategy 及 Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在Fleet
API中，用户可以使用\ ``fleet.DistributedStrategy()``\ 接口定义自己想要使用的分布式策略。

其中\ ``a_sync``\ 选项用于定义参数服务器相关的策略，当其被设定为\ ``False``\ 时，分布式训练将在同步的模式下进行。反之，当其被设定成\ ``True``\ 时，分布式训练将在异步的模式下进行。

.. code:: python

    # 定义异步训练
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = True

    # 定义同步训练
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = False

    # 定义Geo异步训练, Geo异步目前只支持SGD优化算法
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = True
    dist_strategy.a_sync_configs = {"k_steps": 100}

    optimizer = paddle.optimizer.SGD(learning_rate=0.0001)
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
    optimizer.minimize(model.loss)


开始训练
~~~~~~~~

完成模型及训练策略以后，我们就可以开始训练模型了。因为在参数服务器模式下会有不同的角色，所以根据不同节点分配不同的任务。

对于服务器节点，首先用\ ``init_server()``\ 接口对其进行初始化，然后启动服务并开始监听由训练节点传来的梯度。

同样对于训练节点，用\ ``init_worker()``\ 接口进行初始化后，
开始执行训练任务。运行\ ``exe.run()``\ 接口开始训练，并得到训练中每一步的损失值。

.. code:: python

    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()
    else:
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())

        fleet.init_worker()

        for epoch_id in range(1):
            reader.start()
            try:
                while True:
                    loss_val = exe.run(program=paddle.static.default_main_program(),
                                       fetch_list=[avg_cost.name])
                    loss_val = np.mean(loss_val)
                    print("TRAIN ---> pass: {} loss: {}\n".format(epoch_id,
                                                                  loss_val))
            except paddle.core.EOFException:
                reader.reset()
    
        fleet.stop_worker()


运行训练脚本
~~~~~~~~~~~~

定义完训练脚本后，我们就可以用\ ``paddle.distributed.launch``\ 模块运行分布式任务了。其中\ ``server_num``\ ,
``worker_num``\ 分别为服务节点和训练节点的数量。在本例中，服务节点有1个，训练节点有两个。

.. code:: sh

    python -m paddle.distributed.launch --server_num=1 --worker_num=2 train.py
