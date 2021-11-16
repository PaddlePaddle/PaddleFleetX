其他（调节资源的配比、增大bs等）
===========================

说明：本章内容仅适用于飞桨静态图分布式。

原理
----

飞桨使用“线程池”模型调度并执行Op算子。在启动GPU计算之前，通常需要CPU的协助，如调度算子执行，然而如果Op算子本身计算时间很小，“线程池”模型下会带来额外的调度开销。根据实践经验，对于CPU任务设置使用的线程数num_threads=2*dev_count时性能较好，对于GPU任务，设置线程数num_threads=4*dev_count时性能较好。注意：线程池不是越大越好。

操作实践
----

用户只需要指定相应的DistributedStrategy()的开关，就可以设置线程数量。

.. code:: python

    strategy = fleet.DistributedStrategy()

    exe_strategy = paddle.static.ExecutionStrategy()
    exe_strategy.num_threads = 3
    strategy.execution_strategy = exe_strategy

上述例子存放在：\ `example/resnet/train_fleet_static_others.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_static_others.py>`_\ 。

假设要运行8卡的任务，那么只需在命令行中执行:

.. code-block:: sh

   python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 train_fleet_static_others.py

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
    INFO 2021-01-19 14:50:52,903 launch_utils.py:472] Local start 8 processes. First process distributed environment info (Only For Debug):
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:20485               |
        |                     PADDLE_TRAINERS_NUM                        8                      |
        |                PADDLE_TRAINER_ENDPOINTS  ... 0.1:23281,127.0.0.1:41983,127.0.0.1:17503|
        |                     FLAGS_selected_gpus                        0                      |
        |                       PADDLE_TRAINER_ID                        0                      |
        +=======================================================================================+
    ...
    W0119 14:51:04.500844 77798 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 9.2
    W0119 14:51:04.506238 77798 device_context.cc:372] device: 0, cuDNN Version: 7.4.
    W0119 14:51:12.378418 77798 fuse_all_reduce_op_pass.cc:79] Find all_reduce operators: 161. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 5.
    [Epoch 0, batch 0] loss: 0.11252, acc1: 0.03125, acc5: 0.06250
    [Epoch 0, batch 5] loss: 0.11252, acc1: 0.03125, acc5: 0.06250
    [Epoch 0, batch 10] loss: 0.11252, acc1: 0.03125, acc5: 0.06250
    [Epoch 0, batch 15] loss: 0.11252, acc1: 0.03125, acc5: 0.06250

需要注意的是，不同飞桨版本，上述信息可能会有所差异。