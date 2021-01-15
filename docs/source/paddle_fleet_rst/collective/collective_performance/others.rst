其他（调节资源的配比、增大bs等）
===========================



原理
----

PaddlePaddle 使用“线程池”模型调度并执行Op，Op在启动GPU计算之前， 
通常需要CPU的协助，然而如果Op本身占用时间很小，“线程池”模型下又会带来额外的调度开销。
根据以往的经验，对于CPU任务，num_threads=2 * dev_count 时性能较好，
对于GPU任务，num_threads=4 * dev_count 时性能较好。注意：线程池不是越大越好。

操作实践
----

用户只需要指定相应的DistributedStrategy()的开关，就可以设置线程数量。

.. code:: python

    strategy = fleet.DistributedStrategy()

    exe_strategy = paddle.static.ExecutionStrategy()
    exe_strategy.num_threads = 3
    strategy.execution_strategy = exe_strategy

基于ResNet50网络的代码：`example/resnet/train_fleet_static_others.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_static_others.py>`_。
