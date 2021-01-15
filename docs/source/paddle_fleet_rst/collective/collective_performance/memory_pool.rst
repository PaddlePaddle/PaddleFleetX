减少显(内)存直接申请释放
------------------

原理
----

Fleet 实现了底层通过改变通信拓扑，实现分层 allreduce。用户只需要指定相应的DistributedStrategy()
的开关，就可以选择不同的通信拓扑。

操作实践
----

减少scope drop和fetch频率，可以减少频繁的变量内存申请、释放和拷贝， 从而提升性能。

.. code:: python

    # 每 10 batch 之后清理一次临时变量
    strategy = fleet.DistributedStrategy()

    exe_strategy = paddle.static.ExecutionStrategy()
    exe_strategy.num_iteration_per_drop_scope = 10
    strategy.execution_strategy = exe_strategy


    # 降低fetch频率，每 30 batch fetch 一次训练输出
    for pass_id in xrange(PASS_NUM):
        batch_id = 0
        while True:
            if batch_id % 30 == 0:
                fetched = exe.run(fetch_list)
            else:
                exe.run([])

基于ResNet50网络的memory_pool代码：`example/resnet/train_fleet_static_memory_pool.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_static_memory_pool.py>`_。