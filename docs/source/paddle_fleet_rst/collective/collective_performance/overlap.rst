通信重叠
------------------

简介
----


操作实践
----

nccl通信器数量 nccl_comm_num 可以加快GPU之间的通信效率，建议单机设置为1，多机设置为2。

.. code:: python

    strategy = fleet.DistributedStrategy()
    strategy.nccl_comm_num = 2
    strategy.sync_nccl_allreduce=False