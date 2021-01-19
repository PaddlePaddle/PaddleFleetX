OP融合（计算，通信）
===========================


计算融合
----

将模型网络中顺序执行的多个OPs进行融合能够减少OP 调度的开销，提升训练速度。目前Fleet 中支持如下3种的OP 融合：

- fuse_all_optimizer_ops：表明是否融合(fuse) 是否融合 optimizer_op，仅对部分 optimizer 可用（SGD、Adam和Momentum）。

- fuse_elewise_add_act_ops：表明是否融合(fuse) elementwise_add_op和activation_op。

- fuse_bn_act_ops：表明是否融合(fuse) batch_norm_op 和 activation_op。

通常使用这些策略都会使整体执行过程更快。


通信融合
----

AllReduce 融合默认情况下会将同一layer中参数的梯度的多个AllReduce操作合并成一个。 比如对于 fc 中有Weight和Bias两个参数，打开该选项之前，需要两次AllReduce操作；打开该选项之后，只用一次AllReduce 操作。这样可以减少梯度同步时的通信耗时。

此外，为支持更大粒度的参数梯度融合，Fleet 提供了以下两个选项，用户可以在训练程序运行前在DistributedStrategy中设置：

- fuse_grad_size_in_MB: 指定每个AllReduce操作的梯度字节数，如该参数等于16 则每次AllReduce调用传输16MB的梯度。 该参数的经验值为总通信量的十分之一。

- fuse_grad_size_in_TFLOPS: 指定每次AllReduce操作的最大层数，即到达该层数就进行AllReduce。如该参数等于50, 则最多每50层做一次 fused AllReduce。

注意： AllReduce融合目前不支持sparse参数梯度。

操作实践
----

.. code:: python
   
    # 计算融合
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.fuse_elewise_add_act_ops = True
    build_strategy.fuse_bn_act_ops = True
    build_strategy.fuse_relu_depthwise_conv = True
    build_strategy.fuse_broadcast_ops = True
    build_strategy.fuse_all_optimizer_ops = True

    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.build_strategy = build_strategy

    # 通信融合
    strategy.fuse_grad_size_in_MB = 16
    strategy._fuse_grad_size_in_TFLOPS = 50
    strategy.fuse_all_reduce_ops=True


上述例子存放在：`example/resnet/train_fleet_static_op_fusion.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_static_op_fusion.py>`_。
假设要运行2卡的任务，那么只需在命令行中执行:

.. code-block:: sh

   fleetrun --gpus=0,1 train_fleet_static_op_fusion.py

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
    WARNING 2021-01-19 14:53:04,943 launch.py:316] Not found distinct arguments and compiled with cuda. Default use collective mode
    launch train in GPU mode
    INFO 2021-01-19 14:53:04,945 launch_utils.py:472] Local start 8 processes. First process distributed environment info (Only For Debug):
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:28355               |
        |                     PADDLE_TRAINERS_NUM                        8                      |
        |                PADDLE_TRAINER_ENDPOINTS  ... 0.1:33653,127.0.0.1:27766,127.0.0.1:16631|
        |                     FLAGS_selected_gpus                        0                      |
        |                       PADDLE_TRAINER_ID                        0                      |
        +=======================================================================================+
    ...
    W0119 14:53:16.871562 68031 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 9.2
    W0119 14:53:16.875859 68031 device_context.cc:372] device: 0, cuDNN Version: 7.4.
    W0119 14:53:25.973377 68031 build_strategy.cc:116] Currently, fuse_broadcast_ops only works under Reduce mode.
    I0119 14:53:27.382609 68031 graph_pattern_detector.cc:101] ---  detected 16 subgraphs
    I0119 14:53:27.390769 68031 graph_pattern_detector.cc:101] ---  detected 16 subgraphs
    W0119 14:53:27.407582 68031 fuse_optimizer_op_pass.cc:207] Find momentum operators : 161, and 161 for dense gradients. To make the speed faster, those optimization are fused during training.
    W0119 14:53:27.436177 68031 fuse_all_reduce_op_pass.cc:79] Find all_reduce operators: 161. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 6.
    [Epoch 0, batch 0] loss: 0.15131, acc1: 0.00000, acc5: 0.03125
    [Epoch 0, batch 5] loss: 1.15416, acc1: 0.00000, acc5: 0.03125
