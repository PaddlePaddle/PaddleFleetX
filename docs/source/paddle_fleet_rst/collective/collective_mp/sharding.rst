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
该策略的目的是通过 \ ``限制sharding 通信的节点数``和 \ ``增加多路数据并行``来提高训练吞吐。 如果一个模型在普通Sharding 训练时需要M 张GPU，则则开启hybrid-dp 至少需要 N*M GPU （N>= 2）。

Sharding-hybrid-dp 适用的场景如下： 

  * 当前有 4个 8 卡v100 节点
  * 目标模型A 在Sharding 训练时至少需要 8卡 v100 （一个完整的8 卡v100节点）
  * 希望利用全部的 4 个节点来加速训练

上述情况如果直接使用全部的 4 个节点 进行普通的sharding 训练， 那么全部的 32 gpus 之间组成一个完整 Sharding parallelism。这样会因为通信瓶颈造成训练速度非常慢：

  * Sharding 中的broadcast 通信 会涉及全部的32 张卡，且为跨节点通信。
  * Sharding 中的 allreduce 通信 会涉及全部的32 张卡，且为跨节点通信。

开启 hybrid-dp 并设置 \ ``sharding_group_size = 8``后， 每个节点内的 8 张卡组成一个完整的 Sharding parallelism，4 个节点构成 4路 hybrid data parallelism：

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

sharding 可以设置以下参数：

**sharding_segment_strategy(float, optional):** 选择sharding 中用来将前向反向program 切segments 的策略。目前可选策略有："segment_broadcast_MB" 和 "segment_anchors"。 segment 是sharding中引入的一个内部概念，目的是用来让通信和计算相互重叠掩盖（overlap）。默认值是 segment_broadcast_MB. 

**segment_broadcast_MB(float, optional):** 根据sharding 广播通信中的参数量来切segments，仅当 sharding_segment_strategy = segment_broadcast_MB时生效。sharding 会在前向和反向中引入参数广播，在该segment 策略下，每当参数广播量达到 “segment_broadcast_MB”时，在program 中切出一个segment。该参数是一个经验值，最优值会受模型大小和网咯拓扑的影响。 默认值是 32. 

**segment_anchors(list):** 根据用户选定的锚点切割 segments，仅当 sharding_segment_strategy = segment_anchors 生效。该策略可以让用户更精确的控制program 的切分，目前还在实验阶段。

**sharding_degree(int, optional):** sharding并行数。 sharding_degree=1 时，sharding 策略会被关闭。 默认值是 8。

**gradient_merge_acc_step(int, optional):** 梯度累积中的累积步数。 gradient_merge_acc_step=1 梯度累积会被关闭。 默认值是 1。

**optimize_offload(bool, optional):** 优化器状态卸载开关。 开启后会将优化器中的状态(moment) 卸载到Host 的内存中，以到达节省GPU 显存、支持更大模型的目的。开启后，优化器状态会在训练的更新阶段经历：预取-计算-卸载（offload）三个阶段，更新阶段耗时会增加。 这个策略需要权衡显存节省量和训练速度，仅推荐在开启梯度累积并且累积步数较大时开启。 因为累积步数较大时，训练中更新阶段的比例将远小于前向&反向阶段， 卸载引入的耗时将不明显。

**dp_degree(int, optional):** 数据并行的路数。 当dp_degree>=2 时，会在内层并行的基础上，再引入dp_degree路 数据并行。用户需要保证 global_world_size = mp_degree * sharding_degree * pp_degree * dp_degree。 默认值是 1。

**mp_degree(int, optional):** [仅在混合并行中使用] megatron 并行数。 mp_degree=1 时，mp 策略会被关闭。 默认值是 1。

**pp_degree(int, optional):** [仅在混合并行中使用] pipeline 并行数。 pp_degree=1 时，pipeline 策略会被关闭。 默认值是 1。

**pp_allreduce_in_optimize(bool, optional):** [仅在混合并行中使用] 在开启pipeline 并行后，将allreduce 操作从反向阶段移动到更新阶段。根据不同的网络拓扑，该选项会影响训练速度，该策略目前还在实验阶段。 默认值是 False。


为了示例代码的简练，下面例子中使用较小 resnet50 模型作为演示。实际训练中，sharding 的目标是通过牺牲训练速度以换取对更大模型的支持，故不适用于 resnet50 等单卡就能训练的模型。

因为resnet50 较小，我们可以令\ ``sharding_degree = 2``\ 让模型参数被切分为2 个shards，然后在 一个单机4卡 v100 的节点上组成 2 路 dp 并行进行演示。

.. code:: python

    strategy = fleet.DistributedStrategy()
    strategy.sharding = True
    strategy.sharding_configs = {
        "sharding_segment_strategy": "segment_broadcast_MB",
        "segment_broadcast_MB": 32,
        "sharding_degree": 2,
        "dp_degree": 2,
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
    2021-05-12 12:02:20 INFO     Hybrid DP mode turn on !
    2021-05-12 12:02:20 INFO     global word size: 4
    2021-05-12 12:02:20 INFO     global rank: 0
    2021-05-12 12:02:20 INFO     global endpoints: ['127.0.0.1:10033', '127.0.0.1:21161', '127.0.0.1:13997', '127.0.0.1:27877']
    2021-05-12 12:02:20 INFO     global ring id: 3
    2021-05-12 12:02:20 INFO     ##############################
    2021-05-12 12:02:20 INFO     mp group size: 1
    2021-05-12 12:02:20 INFO     mp rank: -1
    2021-05-12 12:02:20 INFO     mp group id: -1
    2021-05-12 12:02:20 INFO     mp group endpoints: []
    2021-05-12 12:02:20 INFO     mp ring id: -1
    2021-05-12 12:02:20 INFO     ##############################
    2021-05-12 12:02:20 INFO     sharding group size: 2
    2021-05-12 12:02:20 INFO     sharding rank: 0
    2021-05-12 12:02:20 INFO     sharding group id: 0
    2021-05-12 12:02:20 INFO     sharding group endpoints: ['127.0.0.1:10033', '127.0.0.1:21161']
    2021-05-12 12:02:20 INFO     sharding ring id: 1
    2021-05-12 12:02:20 INFO     ##############################
    2021-05-12 12:02:20 INFO     pp group size: 1
    2021-05-12 12:02:20 INFO     pp rank: -1
    2021-05-12 12:02:20 INFO     pp group id: -1
    2021-05-12 12:02:20 INFO     pp group endpoints: []
    2021-05-12 12:02:20 INFO     pp ring id: -1
    2021-05-12 12:02:20 INFO     ##############################
    2021-05-12 12:02:20 INFO     pure dp group size: 2
    2021-05-12 12:02:20 INFO     pure dp rank: 0
    2021-05-12 12:02:20 INFO     pure dp group endpoints: ['127.0.0.1:10033', '127.0.0.1:13997']
    2021-05-12 12:02:20 INFO     pure dp ring id: 2
    2021-05-12 12:02:20 INFO     ##############################
    ...
    +==============================================================================+
    |                      sharding=True <-> sharding_configs                      |
    +------------------------------------------------------------------------------+
    |             sharding_segment_strategy           segment_broadcast_MB         |
    |                  segment_broadcast_MB                   32.0                 |
    |                       sharding_degree                    2                   |
    |                             mp_degree                    1                   |
    |                             dp_degree                    2                   |
    |                             hybrid_dp                  False                 |
    |               gradient_merge_acc_step                    1                   |
    |                      optimize_offload                  False                 |
    |              pp_allreduce_in_optimize                  False                 |
    |                             pp_degree                    1                   |
    +==============================================================================+
    ...
    W0114 18:07:51.588716 16234 device_context.cc:346] Please NOTE: device: 4, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.0
    W0114 18:07:51.593963 16234 device_context.cc:356] device: 4, cuDNN Version: 7.6.
    [Epoch 0, batch 0] loss: 4.58475, acc1: 0.03125, acc5: 0.18750
    [Epoch 0, batch 5] loss: 23.57863, acc1: 0.06250, acc5: 0.06250
    [Epoch 0, batch 10] loss: 13.08259, acc1: 0.00000, acc5: 0.06250
    [Epoch 0, batch 15] loss: 9.19330, acc1: 0.00000, acc5: 0.06250
    [Epoch 0, batch 20] loss: 7.46575, acc1: 0.03125, acc5: 0.06250
    [Epoch 0, batch 25] loss: 4.44061, acc1: 0.15625, acc5: 0.18750
    [Epoch 0, batch 30] loss: 5.20638, acc1: 0.06250, acc5: 0.12500
    [Epoch 0, batch 35] loss: 4.75518, acc1: 0.03125, acc5: 0.09375
    [Epoch 0, batch 40] loss: 5.02654, acc1: 0.06250, acc5: 0.09375
    ...


完整4卡的日志信息也可在\ ``./log/``\ 目录下查看。了解更多\ ``fleetrun``\ 的用法可参考左侧文档\ ``fleetrun 启动分布式任务``\ 。


进阶用法
~~~~~~~~~

上面例子介绍了静态图 sharding 的基本用法，能直接应用于 resnet、 transformer 等常见组网组网。 如果用户的组网比较特殊或希望修改sharding 的逻辑可以阅读下面内容。


Sharding 通信组
^^^^^^^^^^^^^^^^
Sharding 会自动每一个Rank（GPU）创建其通信所需的资源 ———— 通信组（groups）， 在Paddle 静态图中每一个通信组都有一个唯一 ring_id 标识。
Sharding 会为每一个 Rank 创建两个通信组：

  * Sharding 通信组（必须）：ring_id=1, group_size = sharding_degree
  * DP 通信组（仅当开启sharidng-dp 时）：ring_id=2, group_size = dp_degree


例如在上文 sharding_degree = 2， dp_degree = 2 的例子中， rank0 上的两个通信组为：

  * Sharding 通信组：ring_id=1, group_size = 2，组成员为[rank0, rank1]
  * DP 通信组：ring_id=2, group_size = 2, 组成员为[rank0, rank3]

用户也可以从训练开始前打印的日志信息中看到对应的信息。 **如果用户希望在模型中引入新的通信组， 需要避免sharding已经占用的 ring_id （1 和 2）。**


Sharding 通信Ops
^^^^^^^^^^^^^^^^^^

通信组建立好后，Sharding 会向模型的前向、反向组网中插入同步通信ops （broadcast）。 用户可以通过打印 Sharidng 生效后生成的 `Program <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/Program_cn.html#program>`__ 查看 Sharidng 通信ops 具体插入的位置。

**同步通信操作的乱序（各rank 间同步通信op插入/执行的顺序的不匹配）非常容易造成训练 hang死或计算错误，所以用户组网中如果希望引入自定义通信op，需要主动避免和原有Sharding 通信ops 产生乱序。**

Sharidng 通信op 的插入逻辑建立在每个rank 相同的组网之上（因为Sharding 本质也是数据并行），并在每一rank上执行相同的插入规则（因为同步通信）， 不会和组网中已存在的用户自定义通信ops 产生组网的“插入乱序”。

“执行乱序”的情况比较特殊，会涉及到模型具体执行逻辑和调度方式。Sharding 中的调度方式是将Sharding 通信ops 和模型计算ops 分别调度到不同的stream上，让通信和计算能最大程度重叠。 一个简单但不太高效的方法是在模型组网里的自定义通信ops 的前后，插入强制的同步， 避免执行时的通信乱序。Paddle 静态图中提供了两个强制同步 op：

  * `c_sync_comm_stream <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/collective/c_sync_comm_stream_op.cc>`__: 同步通信流
  * `c_sync_calc_stream <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/collective/c_sync_calc_stream_op.cc>`__: 同步计算流

用户可以也尝试使用 `wait op <https://github.com/PaddlePaddle/Paddle/pull/31463>`__ 做更进阶的同步和重叠。


