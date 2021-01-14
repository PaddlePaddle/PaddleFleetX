OP融合（计算，通信）
------------------

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


基于ResNet50网络的融合代码：`example/resnet <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_static_op_fusion.py>`_。

