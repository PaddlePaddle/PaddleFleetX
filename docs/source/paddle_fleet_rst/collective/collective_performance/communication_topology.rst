通信拓扑优化
===========================


原理
----

-  TBA

操作实践
----

Fleet 实现了底层通过改变通信拓扑，实现分层 allreduce。用户只需要指定相应的DistributedStrategy()
的开关，就可以选择不同的通信拓扑。

.. code:: python

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.use_hierarchical_allreduce = True
    dist_strategy.hierarchical_allreduce_inter_nranks = 8

上述例子存放在：`example/resnet/train_fleet_static_communication_topology.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_static_communication_topology.py>`_。
假设要运行8卡的任务，那么只需在命令行中执行:

.. code-block:: sh

   fleetrun --gpus=0,1,2,3,4,5,6,7 train_fleet_static_communication_topology.py

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
    INFO 2021-01-19 14:58:43,720 launch_utils.py:472] Local start 8 processes. First process distributed environment info (Only For Debug):
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:53762               |
        |                     PADDLE_TRAINERS_NUM                        8                      |
        |                PADDLE_TRAINER_ENDPOINTS  ... 0.1:58938,127.0.0.1:54203,127.0.0.1:44221|
        |                     FLAGS_selected_gpus                        0                      |
        |                       PADDLE_TRAINER_ID                        0                      |
        +=======================================================================================+
    ...
    W0119 14:58:52.487838 95116 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 9.2
    W0119 14:58:52.493592 95116 device_context.cc:372] device: 0, cuDNN Version: 7.4.
    W0119 14:59:01.665702 95116 fuse_all_reduce_op_pass.cc:79] Find all_reduce operators: 161. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 5.
    [Epoch 0, batch 0] loss: 0.13468, acc1: 0.00000, acc5: 0.06250
    [Epoch 0, batch 5] loss: 0.18902, acc1: 0.03125, acc5: 0.03125