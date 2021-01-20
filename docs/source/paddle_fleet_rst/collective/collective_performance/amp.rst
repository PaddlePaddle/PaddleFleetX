自动混合精度
==========================

简介
----
在使用数据并行分布式训练的同时, 我们还可以引入自动混合精度(Auto Mixed Precision) 来进一步提升训练的速度.

主流的神经网络模型通常使用单精度 ``single-precision`` ``(FP32)``
数据格式来存储模型参数、进行训练和预测. 在上述环节中使用半精度
``half-precision`` ``(FP16)``\ 来代替单精度. 可以带来以下好处:

1. 减少对GPU memory 的需求: GPU 显存不变情况下, 支持更大模型 / batch
   size
2. 降低显存读写时的带宽压力
3. 加速GPU 数学运算速度 (需要GPU
   支持\ `[1] <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensorop>`__)
4. GPU上 FP16 吞吐是FP32 的 2 - 8
   倍\ `[2] <https://arxiv.org/abs/1710.03740>`__

Paddle 支持自动混合精度计算, 并实现了 ``自动维护FP32 、FP16参数副本``,
``Dynamic loss scaling``, ``op黑白名单`` 等策略来避免
因 FP16 动态范围较小而带来的模型最终精度损失。 Fleet 作为Paddle通用的分布式训练API提供了简单易用的接口, 用户只需要添加几行代码
就可将自动混合精度应用到原有的分布式训练中进一步提升训练速度.


原理
----

-  TBA


操作实践
----

Fleet 将AMP 实现为 meta optimizer, 用户需要指定其的
``inner-optimizer``. Fleet AMP支持所有 paddle optimziers 和 FLeet meta
otpimizers 作为其 inner-optimizer。只需要在reset网络基础上打开相应的开关和配置相应的选项。

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

   fleetrun --gpus=0,1,2,3,4,5,6,7 train_fleet_static_amp.py

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
