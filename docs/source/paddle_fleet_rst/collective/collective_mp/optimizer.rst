使用LARS / LAMB 优化分布式超大batch 训练
---------------------------------------

简介
~~~~~

在数据并行分布式训练场景中, 常使用增加GPU数量的方式来加速训练。
为了保证GPU的算力得到充分利用, 每张GPU卡上的batch size都需要足够大。
因此在增加GPU 数量同时, 训练的全局batch size 也会变大。

但越大的全局batch size
会带来训练的收敛问题\ `[1] <https://arxiv.org/abs/1404.5997>`__
`[2] <https://arxiv.org/abs/1609.04836>`__:

-  模型最终精度损失
-  收敛速度变慢, 需要更多的epoch 才能收敛

LARS\ `[3] <https://arxiv.org/abs/1708.03888>`__ 和
LAMB\ `[4] <https://arxiv.org/abs/1904.00962>`__
两个优化策略常用来解决上述超大batch 训练中的收敛问题。

Paddle 实现了这两种优化策略，paddle.distributed.fleet 作为Paddle通用的分布式训练API提供了简单易用的接口, 用户只需要添加几行代码就可将策略加入到原有的训练中。 通过这两个优化策略,
我们在超大batch 场景中实现了更快的收敛速度和无损的精度, 结合Fleet
中其他的策略(e.g. `AMP <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/fleet_collective_training_speedup_with_amp_cn.html>`__)
可以缩短整体训练收敛时间。


原理
~~~~~

LARS
^^^^^^

LARS 公式如下：

.. math::

    local\_learning\_rate = learning\_rate * lars\_coeff * 
            \frac{||param||}{||gradient|| + lars\_weight\_decay * ||param||} 

.. math::
    velocity = mu * velocity + local\_learning\_rate * (gradient + lars\_weight\_decay * param + epsilon) \\

.. math::
    param = param - velocity \\

可以看到LARS 其实是在 带\ ``weight decay`` 的\ ``momentum``
优化器的基础上加入了\ ``local learning rate`` 的逻辑,
对每一层的\ ``learning rate`` 进行了放缩。


LAMB
^^^^^^

LAMB 公式如下：

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
进行了放缩。


效果
~~~~~

使用 LARS 在超大batch size 下训练 resnet50：

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


使用方法
~~~~~~~~~

LARS
^^^^^^

fleet 将 LARS实现为一个 fleet
meta optimizer, 在使用时需要设置以下几点:

1. LARS meta optimizer 的 inner optimizer 必须为 ``momentum``, 并在
   momentum 中定义 ``mu`` 和\ ``lr`` 参数。
2. 在DistributedStrategy 中设置LARS 特有的 ``lars_coeff`` 参数和
   ``lars_weight_decay`` 参数。

   -  LARS 已经将 ``weight decay`` 包含进公式中, 用户不需要再在
      optimizer中设置 ``regularization``。
   -  fleet 中还提供 lars\_weight\_decay 过滤策略,
      可以通过在\ ``exclude_from_weight_decay`` 参数加入对应layer 的
      ``name string``, 让这一 layer 的参数不进行 lars weight decay。
      (通常我们将\ ``BN`` 参数 和\ ``FC_bias`` 从lars weight decay 中过滤)

.. code:: python

    strategy = fleet.DistributedStrategy()
    strategy.lars = True
    strategy.lars_configs = {
                        "lars_coeff": 0.001,
                        "lars_weight_decay": 0.0005,
                        "exclude_from_weight_decay": ['batch_norm', '.b_0']
                    }

上述例子的完整代码存放在：\ `train_fleet_lars.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_lars.py>`_\ 下面。假设要运行2卡的任务，那么只需在命令行中执行:


.. code-block:: sh

   python -m paddle.distributed.launch --gpus=0,1 train_fleet_lars.py

您将看到显示如下日志信息：

.. code-block::

    -----------  Configuration Arguments -----------
    gpus: 0,1
    heter_worker_num: None
    heter_workers: 
    http_port: None
    ips: 127.0.0.1
    log_dir: log
    ...   
    ------------------------------------------------
    ...    
    +=======================================================================================+
    |                        Distributed Envs                      Value                    |
    +---------------------------------------------------------------------------------------+
    |                       PADDLE_TRAINER_ID                        0                      |
    |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:12464               |
    |                     PADDLE_TRAINERS_NUM                        2                      |
    |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:12464,127.0.0.1:43227       |
    |                     FLAGS_selected_gpus                        0                      |
    +=======================================================================================+
    ...
    +==============================================================================+
    |                                                                              |
    |                         DistributedStrategy Overview                         |
    |                                                                              |
    +==============================================================================+
    |                          lars=True <-> lars_configs                          |
    +------------------------------------------------------------------------------+
    |                            lars_coeff          0.0010000000474974513         |
    |                     lars_weight_decay          0.0005000000237487257         |
    |                               epsilon                   0.0                  |
    |             exclude_from_weight_decay                batch_norm              |
    |                                                         .b_0                 |
    +==============================================================================+
    ...
    W0114 18:07:51.588716 16234 device_context.cc:346] Please NOTE: device: 4, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.0
    W0114 18:07:51.593963 16234 device_context.cc:356] device: 4, cuDNN Version: 7.6.
    [Epoch 0, batch 0] loss: 0.14651, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 5] loss: 1.82926, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 10] loss: 0.00000, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 15] loss: 0.13787, acc1: 0.03125, acc5: 0.03125
    [Epoch 0, batch 20] loss: 0.12400, acc1: 0.03125, acc5: 0.06250
    [Epoch 0, batch 25] loss: 0.17749, acc1: 0.00000, acc5: 0.00000
    ...


完整 2卡的日志信息也可在\ ``./log/``\ 目录下查看。


LAMB
^^^^^^

fleet 将 LAMB实现为一个 fleet
meta optimizer, 在使用时需要设置以下几点:

1. LAMB meta optimizer 的 inner optimizer 必须为 ``adam``, 并在 adam
   中定义 学习率\ ``lr``, 一阶 moment 的指数衰减率\ ``beta1``
   和二阶moment 的指数衰减率\ ``beta2`` 参数。
2. 在 DistributedStrategy 里定设置AMB 特有的 ``lamb_weight_decay`` 参数.

   -  LAMB 已经将 ``weight decay`` 包含进公式中, 用户不需要再在
      optimizer中设置 ``regularization``。
   -  fleet 中还提供 lamb\_weight\_decay 过滤策略,
      可以通过在\ ``exclude_from_weight_decay`` 参数加入对应layer 的
      ``name string``, 让这一 layer 的参数不进行 lars weight decay。
      (通常我们将\ ``LN`` 从lamb weight decay 中过滤)

.. code:: python

    strategy = fleet.DistributedStrategy()
    strategy.lamb = True
    strategy.lamb_configs = {
        'lamb_weight_decay': 0.01,
        'exclude_from_weight_decay': ['layer_norm'],
    }

上述例子的完整代码存放在：\ `train_fleet_lamb.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_lamb.py>`_\ 下面。假设要运行2卡的任务，那么只需在命令行中执行:


.. code-block:: sh

   python -m paddle.distributed.launch --gpus=0,1 train_fleet_lamb.py

您将看到显示如下日志信息：

.. code-block::

    -----------  Configuration Arguments -----------
    gpus: 0,1
    heter_worker_num: None
    heter_workers: 
    http_port: None
    ips: 127.0.0.1
    log_dir: log
    ...   
    ------------------------------------------------
    ...    
    +=======================================================================================+
    |                        Distributed Envs                      Value                    |
    +---------------------------------------------------------------------------------------+
    |                       PADDLE_TRAINER_ID                        0                      |
    |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:12464               |
    |                     PADDLE_TRAINERS_NUM                        2                      |
    |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:12464,127.0.0.1:43227       |
    |                     FLAGS_selected_gpus                        0                      |
    +=======================================================================================+
    ...
    +==============================================================================+
    |                                                                              |
    |                         DistributedStrategy Overview                         |
    |                                                                              |
    +==============================================================================+
    |                          lamb=True <-> lamb_configs                          |
    +------------------------------------------------------------------------------+
    |                     lamb_weight_decay           0.009999999776482582         |
    |             exclude_from_weight_decay                layer_norm              |
    +==============================================================================+
    ...
    W0114 18:07:51.588716 16234 device_context.cc:346] Please NOTE: device: 4, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.0
    W0114 18:07:51.593963 16234 device_context.cc:356] device: 4, cuDNN Version: 7.6.
    [Epoch 0, batch 0] loss: 0.14651, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 5] loss: 1.82926, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 10] loss: 0.00000, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 15] loss: 0.13787, acc1: 0.03125, acc5: 0.03125
    [Epoch 0, batch 20] loss: 0.12400, acc1: 0.03125, acc5: 0.06250
    [Epoch 0, batch 25] loss: 0.17749, acc1: 0.00000, acc5: 0.00000
    ...


完整2 卡的日志信息也可在\ ``./log/``\ 目录下查看。
