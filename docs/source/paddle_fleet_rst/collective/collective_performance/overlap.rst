通信重叠
===========================


简介
----

Paddle的通信进行重叠（overlap），可以有效提升通信效率。


原理介绍
----

Paddle的整体框架目前只有一个计算流，但可以有多个通信流。在通信为瓶颈的低配网络中，通过
重叠通信流，可以有效利用通信带宽，从而达到更优的通信性能。多流相关的概念请参考：
`cuda-streams-best-practices <https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf>`_。

使用方法
----

Fleet已经实现通信流overlap，只需设置通信器数量 nccl_comm_num 可以加快GPU之间的通信效率，建议单机设置为1，多机设置为2。

.. code:: python

    strategy = fleet.DistributedStrategy()
    strategy.nccl_comm_num = 2
    strategy.sync_nccl_allreduce=False


上述例子存放在：\ `example/resnet/train_fleet_static_overlap.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_static_overlap.py>`_\ 下面，
假设要运行2卡的任务，那么只需在命令行中执行:

.. code-block:: sh

   fleetrun --gpus=0,1 train_fleet_static_overlap.py

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
    ...
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:10097               |
        |                     PADDLE_TRAINERS_NUM                        2                      |
        |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:10097,127.0.0.1:59371       |
        |                     FLAGS_selected_gpus                        0                      |
        |                       PADDLE_TRAINER_ID                        0                      |
        +=======================================================================================+
    ...
    W0118 21:44:34.542804 70071 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 9.2
    W0118 21:44:34.547377 70071 device_context.cc:372] device: 0, cuDNN Version: 7.4.
    W0118 21:44:40.178053 70071 fuse_all_reduce_op_pass.cc:79] Find all_reduce operators: 161. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 5.
    [Epoch 0, batch 0] loss: 0.14466, acc1: 0.00000, acc5: 0.03125
    [Epoch 0, batch 5] loss: 4.00225, acc1: 0.00000, acc5: 0.03125
    ...
