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

基于ResNet50网络的communication_topology代码：`example/resnet/train_fleet_static_communication_topology.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_static_communication_topology.py>`_。