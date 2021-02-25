流式训练
=====================

简介
---------------------
飞桨参数服务器训练支持流式训练模式，支持配置千亿级大规模稀疏及[0, INT64]范围内的ID映射，支持模型自增长及配置特征准入（不存在的特征可以以适当的条件创建）、淘汰（够以一定的策略进行过期的特征的清理）等策略，支持模型增量保存，通过多种优化来保证流式训练的流程及效果。


原理介绍
---------------------
流式训练(OnlineLearning)， 即训练数据不是一次性放入训练系统，而是随着时间流式的加入到训练过程中去。 整个训练服务不停止，数据经过预处理后进入训练系统参与训练并产出线上所需的预测模型参数。通过流式数据的生产、实时训练及快速部署上线来提升推荐系统的性能和效果。流式训练是按照一定顺序进行数据的接收和处理，每接收一个数据，模型会对它进行预测并对当前模型进行更新，然后处理下一个数据。 像信息流、小视频、电商等场景，每天都会新增大量的数据， 让每天(每一刻)新增的数据基于上一天(上一刻)的模型进行新的预测和模型更新。


功能效果
---------------------
通过合理配置，可实现大规模流式训练，提升推荐系统的性能和效果。

本文中涉及到的相关功能和使用示例：

- 使用大规模稀疏的算子进行组网
- 配置准入策略
- 配置模型保存及增量保存


使用方法
---------------------

大规模稀疏及准入配置
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: paddle.static.nn.sparse_embedding(input, size, padding_idx=None, is_test=False, entry=None, param_attr=None, dtype='float32')

飞桨参数服务器模式使用大规模稀疏需要使用`paddle.static.nn.sparse_embedding`作为embedding lookup层的算子， 而不是使用 `paddle.nn.functional.embedding`。
`paddle.static.nn.sparse_embedding` 采用稀疏模式进行梯度的计算和更新，支持配置训练模式(训练/预测)，支持配置准入策略，且输入接受[0, INT64]范围内的特征ID,  更加符合在线学习的功能需求。


**参数：**

    - input, (Tensor): 存储特征ID的Tensor，数据类型必须为：int32/int64，特征的范围在[0, INT64]之间，超过范围在运行时会提示错误。
    - size, (list|tuple): 形状为(num_embeddings, embedding_dim), 大规模稀疏场景下， 参数规模初始为0，会随着训练的进行逐步扩展，因此num_embeddings 暂时无用，可以随意填一个整数，embedding_dim 则为词嵌入权重参数的维度配置。
    - padding_idx, (int)，如果配置了padding_idx，那么在训练过程中遇>到此id时会被用0填充。
    - is_test, (bool)，训练/预测模式，在预测模式(is_test=False)下，遇到不存在的特征，不会初始化及创建，直接以0填充后返回。
    - entry, (ProbabilityEntry|CountFilterEntry, optinal)，准入策略配置，目前支持概率准入和频次准入。
    - param_attr, (ParamAttr, optinal)，embedding层参数属性，类型是ParamAttr或者None， 默认为None。
    - dtype, (float32|float64, optinal)，输出Tensor的数据类型，支持float32、float64。当该参数值为None时， 输出Tensor的数据类型为float32。默认值为None。

**用法示例：**

    .. code:: python

        ...

        import paddle

        sparse_feature_dim = 1024
        embedding_size = 64

        # 训练过程中，出现超过10次及以上的特征才会参与训练
        entry = paddle.distributed.CountFilterEntry(10)

        input = paddle.static.data(name='ins', shape=[1], dtype='int64')

        emb = paddle.static.nn.sparse_embedding((
            input=input,
            size=[sparse_feature_dim, embedding_size],
            is_test=False,
            entry=entry,
            param_attr=paddle.ParamAttr(name="SparseFeatFactors",
                    initializer=paddle.nn.initializer.Uniform()))


淘汰配置
~~~~~~~~~

.. py:function:: paddle.distributed.fleet.shrink(threshold)

使用此接口，可以按照一定的频率进行过期ID特征的清理，稀疏参数在初始化的时候，会在内部设定一个最近出现时间戳的记录(相对值计数，非timestamp)，当特征ID在训练中出现时，此值被置位0，当用户显示调用`paddle.distributed.fleet.shrink(threshold)`是，此值会主动递增1，当值超过`threshold`时，则会被清理掉。


**参数：**

    - threshold, (int): 对于超过一定时间未出现的特征进行清理。

    .. code:: python

        ...

        import paddle

        ...
        dataset, hour, day = get_ready_training_dataset()

        do_training ...

        # 天级别的淘汰，每天的数据训练结束后，对所有特征的过期时间+1，对超过30天未出现的特征进行清理
        unseen_days = 30

        if fleet.is_first_worker() and hour == 23:
            paddle.distributed.fleet.shrink(unseen_days)



保存及增量保存配置
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: paddle.distributed.fleet.save_persistables(exe, dirname, mode)

模型保存接口，使用该接口会将当前训练中涉及到的模型权重，优化器的中间值全量保存下来，供增量训练、恢复训练、在线预测使用。
针对大规模稀疏，会提供对应的save_base、save_delta等增量保存方案，降低模型保存的磁盘占用及耗时。


**参数：**

    - executor, (Executor): 用于保存持久性变量的 ``executor``。
    - dirname, (str): 用于储存持久性变量的文件目录。`
    - mode, (0|1|2, optinal)，仅支持 `0、1、2` 三个数值的配置，`0` 表示全量保存，`1` 表示保存base模型， `2`表示保存增量模型。


    .. code:: python

        ...

        import paddle

        ...
        dataset, hour, day = get_ready_training_dataset()

        do_training ...

        # 天级别的淘汰，每天的数据训练结束后，对所有特征的过期时间+1，对超过30天未出现的特征进行清理
        unseen_days = 30

        if fleet.is_first_worker() and hour == 0:
            # 每天的0点，保存一次全量模型
            if hour == 0:
                fleet.save_persistables(exe, "output/epoch_{}".format(day), 1)

            # 其他时间点，每个小时保存一次增量模型
            else:
                fleet.save_persistables(exe, "output/epoch_{}".format(day), 2)


常规训练流程
~~~~~~~~~~~~~~~~~~~~~

流式训练是个上下游牵涉众多的训练方法，本文只贴出训练相关的配置给用户做一个讲解，具体使用需要结合实际情况进行代码的伪代码：

.. code-block:: python

    # 初始化分布式环境
    fleet.init()

    # your real net function
    model = net()

    # 使用参数服务器异步训练模式
    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.a_sync = True

    # 分布式训练图优化
    adam = paddle.optimizer.Adam(learning_rate=5e-06)
    adam = fleet.distributed_optimizer(adam, strategy=strategy)
    adam.minimize(model.avg_cost)

    # 启动PServer
    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()

    if fleet.is_worker():
        # 初始化Worker
        exe.run(paddle.static.default_startup_program())
        fleet.init_worker()

        while True:

            # 持续不断的从`get_ready_training_set`获取可训练的书记集和相关的配置
            # 下面是一个按小时训练的例子
            dataset, hour, day = get_ready_training_dataset()

            if dataset is None:
                break

            # 使用`dataset`中的数据进行训练和模型保存
            exe.train_from_dataset(program=paddle.static.default_main_program(),
                                   dataset=dataset,
                                   fetch_list=[model.auc],
                                   fetch_info=["avg_auc"],
                                   print_period=10)

            # 0号保存模型即可，每天第0个小时进行全量保存， 剩余时间进行增量保存
            if fleet.is_first_worker():
                unseen_days = 30

                if hour == 23:
                    paddle.distributed.fleet.shrink(unseen_days)

                if hour == 0:
                    fleet.save_persistables(exe, "output/epoch_{}".format(day), 1)
                else:
                    fleet.save_persistables(exe, "output/epoch_{}".format(day), 2)

        fleet.stop_worker()



运行成功提示
---------------------
[略]


常见问题与注意事项
---------------------
1. 训练过程中，如需使用分布式指标，请参考<分布式指标章节>。
2. 如果训练中途中断，需要加载模型后继续训练，请参考<增量训练章节>


论文/引用
---------------------
[略]

