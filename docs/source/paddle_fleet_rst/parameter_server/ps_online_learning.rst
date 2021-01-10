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
1. 使用大规模稀疏的算子进行组网
2. 配置准入策略
3. 


使用方法
---------------------
流式训练是个上下游牵涉众多的训练方法，本文只贴出训练相关的配置给用户做一个讲解，具体使用需要结合实际情况进行代码的优化：


```
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    fleet.init()

    model = "your real net function" 

    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.a_sync = True

    adam = paddle.fluid.optimizer.Adam(learning_rate=5e-06)
    adam = fleet.distributed_optimizer(adam, strategy=strategy)
    adam.minimize(model.avg_cost)

    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()
    else:
        exe.run(paddle.static.default_startup_program())
        fleet.init_worker()

        datasets = []

        days = ["20200827"]

        all_files = 0
        train_path = "train_data"

        for day_index in range(len(days)):
            day = days[day_index]
            files = "your real training files" 
            dataset = create_dataset(use_var, files)
            datasets.append(dataset)

        for day_index in range(len(days)):
            begin = time.time()
            day = days[day_index]
            dataset = datasets[day_index]
            epochs = config.epoch_num

            for i in range(epochs):
                begin = time.time()
                exe.train_from_dataset(program=paddle.static.default_main_program(),
                                       dataset=dataset,
                                       fetch_list=[model.auc, model.batch_auc],
                                       fetch_info=["avg_auc", "cur_auc"],
                                       print_period=10,
                                       debug=False)
            if fleet.is_first_worker():
                fleet.save_persistables(exe, "output/epoch_{}".format(day_index))

    fleet.stop_worker()

```


运行成功提示
---------------------
[略]


常见问题与注意事项
---------------------
训练过程中，如需使用分布式指标，请参考分布式指标章节。


论文/引用
---------------------
[略]


