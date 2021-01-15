通信重叠
===========================


简介
----

Paddle的通信是可以进行重叠（overlap），从而提升通信效率。


原理介绍
----

Paddle的整体框架目前只有一个计算流，但可以有多个通信流。在通信为瓶颈的低配网路中，通过
重叠通信流，可以有效利用通信带宽，从而达到更优的通信性能。多流相关的概念请参考：
`cuda-streams-best-practices <https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf>`_。

使用方法
----

Fleet已经实现通信流overlap，只需设置通信器数量 nccl_comm_num 可以加快GPU之间的通信效率，建议单机设置为1，多机设置为2。

.. code:: python

    strategy = fleet.DistributedStrategy()
    strategy.nccl_comm_num = 2
    strategy.sync_nccl_allreduce=False


基于ResNet50网络的overlap代码：`example/resnet/train_fleet_static_overlap.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_static_overlap.py>`_。
