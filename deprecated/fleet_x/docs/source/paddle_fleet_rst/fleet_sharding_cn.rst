使用Sharding 训练超大模型
=========================
简介
----

当模型参数达到百亿或者千亿时， 传统的数据并行训练可能会遇到显存瓶颈。 
在数据并行训练中，每个gpu worker 都有一份完整模型参数和优化器状态副本。 
`[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models] <https://arxiv.org/abs/1910.02054>`__
指出在每个GPU 上都保存一份模型参数和优化器状态副本是冗余的。 我们可以通过将上述参数和副本划分到不同GPU 中，
在每个GPU 只保存部分副本，来减少每张GPU上显存的占用，从而可以支持更大模型的训练。 

sharding 实现了类似ZeRO-DP 的训练策略，通过 paddle.distributed.fleet 提供了简单易用的接口, 用户只需要添加几行代码
就可将策略加入到原有的训练中。 

需要注意的是： 目前sharding 策略只在并行GPU 间切分了模型参数和优化器状态（ZeRO-DP）。 模型参数和优化器状态所消耗
的显存将随着并行GPU 数量增加而线性减少。 但每张 GPU 任然保存模型的训练过程中的中间变量（activations），这部分显存消耗
不会随着GPU 数量的增加而减小， 用户可以通过结合 recompute 策略来减少这部分的显存消耗。

通过sharding 和增加并行GPU 数量，用户可以训练百亿或者千亿参数两的超大模型 （可能需要结合 recompute, amp 策略）。 

下文中我们将给出使用 sharding + recompute + amp 在 64卡 v100 上训练 Bert-Giant （27.4 B 模型参数）简单示例。

试验效果
~~~~~~~~
下面表格将对比 sharding 策略对显存影响。 

模型为 Bert-Large，试验环境为 v100 （32GB）， recompute = ON, amp = ON, 

当并行GPU数量增加时，显存的消耗将减小。 

+-----------------------+---------+
| setting               | GPU Mem | 
+=======================+=========+
| sharding—off          | 8667 MB |
+-----------------------+---------+
| sharding—on N1C2      | 5615 MB |
+-----------------------+---------+
| sharding—on N1C4      | 4841 MB |
+-----------------------+---------+
| sharding—on N1C8      | 4127 MB |
+-----------------------+---------+
| sharding—on N2C16     | 3700 MB |
+-----------------------+---------+

Bert-Giant 快速开始
--------------------

添加依赖
~~~~~~~~

添加paddle依赖 和设置环境变量。

.. code:: python

    import os
    import numpy as np
    import fleetx as X
    import paddle
    import paddle.fluid as fluid
    import paddle.distributed.fleet as fleet
    from paddle.distributed.fleet.meta_optimizers.sharding as sharding
    import time
    import sys

    os.environ['FLAGS_enable_parallel_graph'] = "0"
    os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.88"
    os.environ['FLAGS_sync_nccl_allreduce'] = "1"
    os.environ['FLAGS_eager_delete_tensor_gb'] = "0"
    os.environ['FLAGS_fuse_parameter_memory_size'] = "32"
    os.environ['FLAGS_fuse_parameter_groups_size'] = "50"
    os.environ['FLAGS_check_nan_inf'] = "0"

定义分布式模式并初始化
~~~~~~~~~~~~~~~~~~~~

通过\ ``X.parse_train_configs()``\ 接口，用户可以定义训练相关的参数，如：学习率、衰减率等。

同时通过\ ``fleet.init()``\ 接口定义了分布式模型，下面代码中的\ ``is_collective=True``\ 表示采用集合通信的GPU分布式模式训练模型。

.. code:: python

    paddle.enable_static()
    configs = X.parse_train_configs()
    profile = False
    batch_size = 6144
    configs.lr = 1e-4

加载模型及数据
~~~~~~~~~~~~~

用户可以通过\ ``X.applications``\ 接口加载我们预先定义好的模型，如：Resnet50、VGG16、BERT等。并使用定制化的data\_loader加载模型，同时可以定义训练中使用的batch\_size等参数。

.. code:: python

    model = X.applications.BertGiant(lang="en")
    downloader = X.utils.Downloader()
    local_path = downloader.download_from_hdfs('bert.yaml', 'data', fleet.worker_num(), fleet.worker_index())
    data_loader = model.get_train_dataloader(
        data_dir='{}'.format(local_path),
        max_seq_len=512,
        batch_size=batch_size,
        in_tokens=True,
    )


定义分布式及Sharding 相关策略
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于sharding， 用户只需要设置 \ ``fuse_broadcast_MB``\  参数。该参数控制广播通信中参数融合的阈值，会影响sharding 训练中的通信速度，是一个需要根据具体模型大小和网络拓扑设定的经验值。

另外在sharding.utils 提供了两个工具：

1. \ ``comm_analyse``\ : 分析当前模型各层参数大小所在的范围，该数据可以用来帮助设定 fuse_broadcast_MB 参数。
2. \ ``add_sync_comm``\ : 如果需要从训练用的main_prog 中克隆得到 test_prog，需要用这个api 来更新 test_prog， 保证sharding 功能在 test_prog 中被正确添加。

.. code:: python

    place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 2
    exec_strategy.num_iteration_per_drop_scope = 1
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.nccl_comm_num = 3

    dist_strategy.amp = True
    dist_strategy.recompute = True
    dist_strategy.recompute_configs = {"checkpoints": model.checkpoints}
    dist_strategy.sharding_configs = {
    "fuse_broadcast_MB": 32,
    }
    dist_strategy.sharding = True

    scheduled_lr = X.utils.linear_warmup_decay(configs.lr, warmup_steps=4000,
                                                num_train_steps=1000000)
    optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)

    clip_norm_thres = 1.0
    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm_thres))

    optimizer.minimize(model.loss)

    # will print statistic of parameters which will be used in communication.
    sharding.utils.comm_analyse(model.main_prog)

    # clone test_prog if need 
    # when use sharding, test prog clone should be performed after optimizer.minimize(model.loss)
    # model.test_prog = model.main_prog.clone(for_test=True)
    # sharding.utils.add_sync_comm(model.test_prog, dist_strategy)



开始训练
~~~~~~~~~

sharding 训练的模型保存和数据并行训练中的方式略有不同。 因为每张GPU 只保存了部分的模型参数，
需要在每个GPU 进程上都调用 \ ``sharding.utils.save_persistables``\ 接口，将这张GPU上的参数存到GPU所在节点硬盘上的指定目录。 (模型加载方式和数据并行时相同，直接调用paddle.fluid.io.load_persistables 即可)

如果是多节点训练，模型参数将分散在不同节点，用户可以在训练结束后，通过HDFS脚本 等方式上传不同节点上的参数文件。

.. code:: python

    place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    for i, data in enumerate(data_loader()):
        start_time = time.time()
        cost = exe.run(model.main_prog,
                            feed=data,
                            fetch_list=[model.loss.name])

        end_time = time.time()
        print(
            "worker_index: %d, step%d cost = %f, speed: %f"
            % (fleet.worker_index(), i, cost[0], batch_size / (end_time - start_time)))

    # Save model
    # every rank should execute the following function to save its own shard of model params, 
    # all gpus within one node will save their params into the node their belong to, 
    # but if the training is across multiple nodes, we still need to collect the params from each training node
    dirname="/path/to/save_model"  
    sharding.utils.save_persistables(exe, dirname, main_program=model.main_prog, filename=None)


运行训练脚本
~~~~~~~~~~~~

完成上述脚本的编写后，我们就可以使用以下命令一行启动单机多卡分布式训练：
训练Bert-Giant 至少需要 64卡 v100 （32GB）。

.. code:: sh
 
    fleetrun --ips="xx.xx.xx.xx, ..., yy.yy.yy.yy" --log_dir log example_sharding.py
