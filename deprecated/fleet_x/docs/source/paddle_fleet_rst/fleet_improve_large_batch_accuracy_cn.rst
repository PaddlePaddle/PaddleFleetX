使用LARS / LAMB 优化分布式超大batch 训练
========================================

简介
----

在数据并行分布式训练场景中, 常使用增加GPU数量的方式来加速训练.
为了保证GPU的算力得到充分利用, 每张GPU卡上的batch size都需要足够大.
因此在增加GPU 数量同时, 训练的全局batch size 也会变大.

但越大的全局batch size
会带来训练的收敛问题\ `[1] <https://arxiv.org/abs/1404.5997>`__
`[2] <https://arxiv.org/abs/1609.04836>`__:

-  模型最终精度损失
-  收敛速度变慢, 需要更多的epoch 才能收敛

LARS\ `[3] <https://arxiv.org/abs/1708.03888>`__ 和
LAMB\ `[4] <https://arxiv.org/abs/1904.00962>`__
两个优化策略常用来解决上述超大batch 训练中的收敛问题.

Paddle 实现了这两种优化策略，paddle.distributed.fleet 作为Paddle通用的分布式训练API提供了简单易用的接口, 用户只需要添加几行代码
就可将策略加入到原有的训练中。 通过这两个优化策略,
我们在超大batch 场景中实现了更快的收敛速度和无损的精度, 结合Fleet
中其他的策略(e.g. `AMP <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/fleet_collective_training_speedup_with_amp_cn.html>`__)
可以极大缩短的训练整体的time2train.


试验效果
~~~~~~~~


+-----------------------+---------------------+---------+---------+
| resnet50 imagenet     | Global batch size   | epoch   | top1    |
+=======================+=====================+=========+=========+
| [Goyal et al]         | 8k                  | 90      | 76.3%   |
+-----------------------+---------------------+---------+---------+
| LARS Paper            | 32k                 | 90      | 72.3%   |
+-----------------------+---------------------+---------+---------+
| [fleet: lars + amp]   | 16k                 | 60      | 76.2%   |
+-----------------------+---------------------+---------+---------+
| [fleet: lars + amp]   | 32k                 | 62      | 75.9%   |
+-----------------------+---------------------+---------+---------+

LARS
----

我们以在单机多卡上Resent50 训练为简单例子介绍fleet 中 LARS的用法。 

添加依赖
^^^^^^^^

.. code:: python

    import os
    import fleetx as X
    import paddle
    paddle.enable_staic()
    import paddle.fluid as fluid
    import paddle.distributed.fleet.base.role_maker as role_maker
    import time
    import paddle.distributed.fleet as fleet

定义分布式模式并初始化
^^^^^^^^^^^^^^^^^^^^^^

通过\ ``X.parse_train_configs()``\ 接口，用户可以定义训练相关的参数，如：学习率、衰减率等。同时通过\ ``fleet.init()``\ 接口定义了分布式模型，下面代码中的\ ``is_collective=True``\ 表示采用集合通信的GPU分布式模式训练模型。

.. code:: python

    paddle.enable_static()
    configs = X.parse_train_configs()
    fleet.init(is_collective=True)

加载模型及数据
^^^^^^^^^^^^^^

用户可以通过\ ``X.applications``\ 接口加载我们预先定义好的模型，如：Resnet50、VGG16、BERT等。并使用定制化的data\_loader加载模型，同时可以定义训练中使用的batch\_size等参数。

.. code:: python

    model = X.applications.Resnet50()
    downloader = X.utils.Downloader()
    local_path = downloader.download_from_bos(
        fs_yaml='https://fleet.bj.bcebos.com/test/loader/small_imagenet.yaml',
        local_path='./data')
    batch_size = 32
    loader = model.get_train_dataloader(local_path, batch_size=batch_size)

定义分布式及LARS 相关策略
^^^^^^^^^^^^^^^^^^^^^^^^^

LARS 优化算法的公式如下:

.. math::

    local\_learning\_rate = learning\_rate * lars\_coeff * 
            \frac{||param||}{||gradient|| + lars\_weight\_decay * ||param||} 

.. math::
    velocity = mu * velocity + local\_learning\_rate * (gradient + lars\_weight\_decay * param + epsilon) \\

.. math::
    param = param - velocity \\

可以看到LARS 其实是在 带\ ``weight decay`` 的\ ``momentum``
优化器的基础上加入了\ ``local learning rate`` 的逻辑,
对每一层的\ ``learning rate`` 进行了放缩. fleet 将 LARS实现为一个 fleet
meta optimizer, 在使用时需要设置一下几点:

1. LARS meta optimizer 的 inner optimizer 必须为 ``momentum``, 并在
   momentum 中定义 ``mu`` 和\ ``lr`` 参数.
2. 在 fleet dist\_strategy 定义LARS 特有的 ``lars_coeff`` 参数和
   ``lars_weight_decay`` 参数.

   -  LARS 已经将 ``weight decay`` 包含进公式中, 用户不需要再在
      optimizer中设置 ``regularization``.
   -  fleet 中还提供 lars\_weight\_decay 过滤策略,
      可以通过在\ ``exclude_from_weight_decay`` 参数加入对应layer 的
      ``name string``, 让这一 layer 的参数不进行 lars weight decay.
      (通常我们将``BN`` 参数 和 ``FC_bias`` 从lars weight decay 中过滤)

.. code:: python

    dist_strategy = fleet.DistributedStrategy()

    dist_strategy.lars = True
    dist_strategy.lars_configs = {
                        "lars_coeff": 0.001,
                        "lars_weight_decay": 0.0005,
                        "exclude_from_weight_decay": ['batch_norm', '.b_0']
                    }

    optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
    optimizer.minimize(model.loss)

开始训练
^^^^^^^^

这一部分和fleet 中其他任务基本相同:

.. code:: python

    place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    for i, data in enumerate(loader()):
        start_time = time.time()
        cost_val = exe.run(model.main_prog,
                            feed=data,
                            fetch_list=[model.loss.name])

        end_time = time.time()
        print(
            "worker_index: %d, step%d cost = %f, speed: %f"
            % (fleet.worker_index(), i, cost_val[0], batch_size / (end_time - start_time)))

运行训练脚本
~~~~~~~~~~~~

完成上述脚本的编写后，我们就可以使用以下命令一行启动单机多卡分布式训练：

.. code:: sh

    fleetrun --gpus 0,1,2,3,4,5,6,7 --log_dir log example_lars.py

LAMB
----

我们以在单机多卡上Bert 训练为简单例子介绍fleet 中LAMB 的用法.

添加依赖
^^^^^^^^

.. code:: python

    import os
    import fleetx as X
    import paddle
    paddle.enable_staic()
    import paddle.fluid as fluid
    import paddle.distributed.fleet.base.role_maker as role_maker
    import time
    import paddle.distributed.fleet as fleet

定义分布式模式并初始化
^^^^^^^^^^^^^^^^^^^^^^

这一步和上文中的LARS 一致。

.. code:: python

    paddle.enable_static()
    configs = X.parse_train_configs()
    fleet.init(is_collective=True)

加载模型及数据
^^^^^^^^^^^^^^

这一步和上文中的LARS 一致。

.. code:: python

    model = X.applications.Resnet50()
    downloader = X.utils.Downloader()
    local_path = downloader.download_from_bos(
        fs_yaml='https://fleet.bj.bcebos.com/test/loader/small_imagenet.yaml',
        local_path='./data')
    batch_size = 32
    loader = model.get_train_dataloader(local_path, batch_size=batch_size)

定义分布式及LARS 相关策略
^^^^^^^^^^^^^^^^^^^^^^^^^

LAMB 优化算法的公式如下:

.. math::

    m_t = \beta_1 m_{t - 1}+ (1 - \beta_1)g_t \\

.. math::

    v_t = \beta_2 v_{t - 1}  + (1 - \beta_2)g_t^2 \\

.. math::

    r_t = \frac{m_t}{\sqrt{v_t}+\epsilon} \\

.. math::

    w_t = w_{t-1} -\eta_t \frac{\left \| w_{t-1}\right \|}{\left \| r_t + \lambda w_{t-1}\right \|} (r_t + \lambda w_{t-1}) \\

和LARS 类似, LAMB 也是在内层优化器的基础上,
套了一个\ ``local learning rate`` 的逻辑, 对每一层的\ ``learning rate``
进行了放缩. fleet 将 LAMB实现为一个 fleet meta optimizer,
在使用时需要设置一下几点:

1. LAMB meta optimizer 的 inner optimizer 必须为 ``adam``, 并在 adam
   中定义 学习率\ ``lr``, 一阶 moment 的指数衰减率\ ``beta1``
   和二阶moment 的指数衰减率\ ``beta2`` 参数.
2. 在 fleet dist\_strategy 定义LAMB 特有的 ``lamb_weight_decay`` 参数.

   -  LAMB 已经将 ``weight decay`` 包含进公式中, 用户不需要再在
      optimizer中设置 ``regularization``.
   -  fleet 中还提供 lamb\_weight\_decay 过滤策略,
      可以通过在\ ``exclude_from_weight_decay`` 参数加入对应layer 的
      ``name string``, 让这一 layer 的参数不进行 lars weight decay.
      (通常我们将``LN`` 从lamb weight decay 中过滤)

.. code:: python

    dist_strategy = fleet.DistributedStrategy()

    dist_strategy.lamb = True
    dist_strategy.lamb_configs = {
                        'lamb_weight_decay': 0.01,
                        'exclude_from_weight_decay': ['layer_norm'],
                    }

    optimizer = paddle.optimizer.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
    optimizer.minimize(model.loss)

开始训练
^^^^^^^^

这一部分和fleet 中其他任务基本相同:

.. code:: python

    place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    for i, data in enumerate(loader()):
        start_time = time.time()
        cost_val = exe.run(model.main_prog,
                            feed=data,
                            fetch_list=[model.loss.name])

        end_time = time.time()
        print(
            "worker_index: %d, step%d cost = %f, speed: %f"
            % (fleet.worker_index(), i, cost_val[0], batch_size / (end_time - start_time)))

运行训练脚本
~~~~~~~~~~~~

完成上述脚本的编写后，我们就可以使用以下命令一行启动单机多卡分布式训练：

.. code:: sh

    fleetrun --gpus 0,1,2,3,4,5,6,7 --log_dir log resnet50_lamb.py
