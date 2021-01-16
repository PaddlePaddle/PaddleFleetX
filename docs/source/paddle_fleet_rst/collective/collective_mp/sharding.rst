使用Sharding 训练超大模型
-------------------------

简介
~~~~~

当模型参数达到百亿或者千亿时， 传统的数据并行训练可能会遇到显存瓶颈。 
在数据并行训练中，每个gpu worker 都有一份完整模型参数和优化器状态副本。 
`《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》 <https://arxiv.org/abs/1910.02054>`__
指出在每个GPU 上都保存一份模型参数和优化器状态副本是冗余的。 我们可以通过将上述参数和副本划分到不同GPU 中，
在每个GPU 只保存部分副本，来减少每张GPU上显存的占用，从而可以支持更大模型的训练。 


原理
~~~~~

Sharding
^^^^^^^^^^

Sharding 实现了类似ZeRO-DP 的训练策略，将模型参数（parameter）、参数梯度（gradient）、参数对应的优化器状态（moment）切分到每一张GPU 上。让模型参数部分所占的显存随并行卡数的增加而减少。
通过 paddle.distributed.fleet 提供的简单易用接口, 用户只需要添加几行代码就可将策略加入到原有的训练中。 

模型训练过程中的显存消耗主要由两大部分组成：模型参数及优化器状态、训练产生的中间变量（activations）。
sharding 策略只切分了模型参数和优化器状态，因此模型参数和优化器状态所消耗的显存可以随着并行GPU数量增加而线性减少； 
但是每张GPU上仍然维护着模型完整的前向和反向，所以每张GPU依然需要存放模型的训练过程中的产生的全部的中间变量，这部分显存消耗
不会随着GPU 数量的增加而减少。 用户可以通过结合 recompute 策略来减少 activation这部分的显存消耗。

通过sharding 和增加并行GPU 数量，用户可以训练百亿甚至千亿参量的超大模型 （需要结合 recompute, amp 策略）。 

Sharding-hybrid-dp
^^^^^^^^^^^^^^^^^^^^

Sharding hybrid数据并行策略，在sharding 并行的基础上再增加一层数据并行逻辑。
该策略的目的是通过 \ ``限制sharding 通信的节点数`` 和 \ ``增加多路数据并行`` 来提高训练吞吐。 如果一个模型在普通Sharding 训练时需要M 张GPU，则则开启hybrid-dp 至少需要 N*M GPU （N>= 2）。

Sharding-hybrid-dp 适用的场景如下： 

  * 当前有 4个 8 卡v100 节点
  * 目标模型A 在Sharding 训练时至少需要 8卡 v100 （一个完整的8 卡v100节点）
  * 希望利用全部的 4 个节点来加速训练

上述情况如果直接使用全部的 4 个节点 进行普通的sharding 训练， 那么全部的 32 gpus 之间组成一个完整 Sharding parallelism。这样会因为通信瓶颈造成训练速度非常慢：

  * Sharding 中的broadcast 通信 会涉及全部的32 张卡，且为跨节点通信。
  * Sharding 中的 allreduce 通信 会涉及全部的32 张卡，且为跨节点通信。

开启 hybrid-dp 并设置 \ ``sharding_group_size = 8`` 后， 每个节点内的 8 张卡组成一个完整的 Sharding parallelism，4 个节点构成 4路 hybrid data parallelism：

  * Sharding 中的broadcast 通信被限制在每个节点内的 8 张GPU 之间， 没有跨节点通信。
  * Sharding 中的 allreduce 为跨节点通信，但每个allreduce 通信只涉及 对应 sharding_group 上 rank 相同的 4 张GPUs， 且每张GPU仅需要 allreduce通信 1/8 的模型参数。

Sharding-hybrid-dp 通过上述措施，可以较大程度 减少 Sharding 训练 从1节点扩展到4 节点时的（跨节点）通信量。提高节点增加时的加速比，提高训练吞吐。

P.S. hybrid dp 是因为 Sharding parallelism 本身内含一层 data parallelism 逻辑， hybrid dp 是在 Sharding parallelism之上再增加新的一层 data parallelism 逻辑。


效果
~~~~~

下面表格将对比 Sharding 策略对显存的影响。 

模型为 Bert-Large，试验环境为 v100 （32GB）， recompute = ON, amp = ON, hybrid-dp = OFF。
模型不变，单卡batch size 不变，当并行GPU数量增加时，显存的消耗将减小。 省下的显存可以用来增大模型。

+-----------------------+---------+
| setting               | GPU Mem | 
+=======================+=========+
| sharding—off          | 8667 MB |
+-----------------------+---------+
| sharding—on N1C2      | 5615 MB |
+-----------------------+---------+
| sharding—on N1C4      | 4841 MB |
+-----------------------+---------+
| sharding—on N1C8      | 4127 MB |
+-----------------------+---------+
| sharding—on N2C16     | 3700 MB |
+-----------------------+---------+

Sharding 结合 amp + recompute，可以在 128 张 32GB V100 并行的情况下支持千亿参数（115B）ERNIE 训练。



使用方法
~~~~~~~~~

对于sharding，用户需要设置 \ ``fuse_broadcast_MB``\ 参数。该参数控制广播通信中参数融合的阈值，会影响sharding 训练中的通信速度，是一个需要根据具体模型大小和网络拓扑设定的经验值。

若开启hybrid-dp，用户需要设置 \ ``hybrid_dp``\ 为True，并指定 \ ``sharding_group_size``\。 

为了示例代码的简练，下面例子中使用较小 resnet50 模型作为演示。实际训练中，sharding 的目标是通过牺牲训练速度以换取对更大模型的支持，故不适用于 resnet50 等单卡就能训练的模型。

因为resnet50 较小，我们可以令\ ``sharding_group_size = 2``\ 让模型参数被切分为2 个shards，然后在 一个单机4卡 v100 的节点上组成 2 路 hybrid-dp 并行进行演示。

.. code:: python

    strategy = fleet.DistributedStrategy()
    strategy.sharding = True
    strategy.sharding_configs = {
        "fuse_broadcast_MB": 32,
        "hybrid_dp": True,
        "sharding_group_size": 2,
    }


上述例子的完整代码存放在：\ `train_fleet_sharding.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_sharding.py>`_\ 下面。假设要运行4卡的任务，那么只需在命令行中执行:


.. code-block:: sh

   fleetrun --gpus=4,5,6,7 train_fleet_sharding.py


您将看到显示如下日志信息：

.. code-block::

    -----------  Configuration Arguments -----------
    gpus: 4,5,6,7
    heter_worker_num: None
    heter_workers: 
    http_port: None
    ips: 127.0.0.1
    log_dir: log
    ...   
    ------------------------------------------------
    ...    
    +=======================================================================================+
    |                        Distributed Envs                      Value                    |
    +---------------------------------------------------------------------------------------+
    |                       PADDLE_TRAINER_ID                        0                      |
    |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:18362               |
    |                     PADDLE_TRAINERS_NUM                        4                      |
    |                PADDLE_TRAINER_ENDPOINTS  ... 0.1:23911,127.0.0.1:35135,127.0.0.1:38263|
    |                     FLAGS_selected_gpus                        4                      |
    +=======================================================================================+
    ...
    INFO:root:Using Sharing&DP mode !
    INFO:root:global word size: 4
    INFO:root:global rank: 0
    INFO:root:sharding group_size: 2
    INFO:root:sharding rank: 0
    INFO:root:dp group size: 2
    INFO:root:dp rank: 0
    INFO:root:current endpoint: 127.0.0.1:18362
    INFO:root:sharding group endpoints: ['127.0.0.1:18362', '127.0.0.1:23911']
    INFO:root:dp group endpoints: ['127.0.0.1:18362', '127.0.0.1:35135']
    INFO:root:global word endpoints: ['127.0.0.1:18362', '127.0.0.1:23911', '127.0.0.1:35135', '127.0.0.1:38263']
    server not ready, wait 3 sec to retry...
    not ready endpoints:['127.0.0.1:23911']
    ...
    +==============================================================================+
    |                      sharding=True <-> sharding_configs                      |
    +------------------------------------------------------------------------------+
    |                     fuse_broadcast_MB                   32.0                 |
    |                             hybrid_dp                   True                 |
    |                   sharding_group_size                    2                   |
    +==============================================================================+
    ...
    W0114 18:07:51.588716 16234 device_context.cc:346] Please NOTE: device: 4, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.0
    W0114 18:07:51.593963 16234 device_context.cc:356] device: 4, cuDNN Version: 7.6.
    [Epoch 0, batch 0] loss: 0.14651, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 5] loss: 1.82926, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 10] loss: 0.00000, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 15] loss: 0.13787, acc1: 0.03125, acc5: 0.03125
    [Epoch 0, batch 20] loss: 0.12400, acc1: 0.03125, acc5: 0.06250
    [Epoch 0, batch 25] loss: 0.17749, acc1: 0.00000, acc5: 0.00000
    ...


完整4卡的日志信息也可在\ ``./log/``\ 目录下查看。了解更多\ ``fleetrun``\ 的用法可参考左侧文档\ ``fleetrun 启动分布式任务``\ 。
