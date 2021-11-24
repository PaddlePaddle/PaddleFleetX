Gradient Merge
------------------

简介
~~~~~

为了提升模型的性能，人们开始追求：更大规模的数据集、更深的网络层、更庞大的参数规模。但是随之而来的就是给模型训练带来了巨大的压力，因此分布式技术及定制化AI 芯片应运而生。但在分布式训练中，经常会遇到显存或者内存不足的情况，通常是以下几点原因导致的：

-  输入的数据过大，例如视频类训练数据。
-  深度模型的参数过多或过大，所需的存储空间超出了内存/显存的大小。
-  AI芯片的内存有限。

为了能正常完成训练，我们通常只能使用较小的batch
size 以降低模型训练中的所需要的存储空间，这将导致很多模型无法通过提高训练时的batch
size 来提高模型的精度。

Gradient Merge (GM) 策略的主要思想是将连续多个batch 数据训练得到的参数梯度合并做一次更新。
在该训练策略下，虽然从形式上看依然是小batch 规模的数据在训练，但是效果上可以达到多个小batch 数据合并成大batch 后训练的效果。


原理
~~~~~

Gradient Merge 只是在训练流程上做了一些微调，达到模拟出大batch
size 训练效果的目的。具体来说，就是使用若干原有大小的batch 数据进行训练，即通过“前向+反向”
网络计算得到梯度。其间会有一部分显存/内存用于存放梯度，然后对每个batch计算出的梯度进行叠加，当累加的次数达到某个预设值后，使用累加的梯度对模型进行参数更新，从而达到使用大batch 数据训练的效果。

在较大的粒度上看， GM 是将训练一个step 的过程由原来的 “前向 + 反向 + 更新” 改变成 “（前向 + 反向 + 梯度累加）x k + 更新”， 通过在最终更新前进行 k 次梯度的累加模拟出 batch size 扩大 k 倍的效果。 
更具体细节可以参考 `《MG-WFBP: Efficient Data Communication for Distributed Synchronous SGD Algorithms》 <https://arxiv.org/abs/1811.11141>`__  。

静态图使用方法
~~~~~~~~~

Gradient Merge
策略在使用方面也很简单，用户只需要定义将多少batch 的数据计算出的梯度叠加更新模型参数，便可以实现大batch 训练的目的。

训练代码的框架和其他fleet 训练代码基本一样，用户只需要在 fleet.DistributedStrategy 中配置Gradient Merge 相关参数即可。

假设我们定义了batch
size 为 N；通过设置\ ``k_steps``\，使用4个batch
size来模拟一个大batch的训练，从而达到了batch size 为 4*N 的训练效果。

在\ ``gradient_merge_configs``\ 中，avg 选项用于控制梯度累计的形式：当被设置为
True
时，会对每次的梯度求和并做平均；反之将直接对梯度求和，并对参数进行更新。

.. code:: python

   strategy = fleet.DistributedStrategy()
   # 使用Gradient merge策略并设置相关参数
   strategy.gradient_merge = True
   strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}

上述例子的完整代码存放在：\ `train_fleet_gradient_merge.py <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/resnet/train_fleet_gradient_merge.py>`_\ 下面。假设要运行2卡的任务，那么只需在命令行中执行:


.. code-block:: sh

   python -m paddle.distributed.launch --gpus=0,1 train_fleet_gradient_merge.py


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
        |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:17901               |
        |                     PADDLE_TRAINERS_NUM                        2                      |
        |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:17901,127.0.0.1:18846       |
        |                     FLAGS_selected_gpus                        0                      |
        |                       PADDLE_TRAINER_ID                        0                      |
        +=======================================================================================+
    ...
        +==============================================================================+
        |                                                                              |
        |                         DistributedStrategy Overview                         |
        |                                                                              |
        +==============================================================================+
        |                gradient_merge=True <-> gradient_merge_configs                |
        +------------------------------------------------------------------------------+
        |                               k_steps                    4                   |
        |                                   avg                   True                 |
        +==============================================================================+
    ...
    W0104 17:59:19.018365 43338 device_context.cc:342] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 9.2
    W0104 17:59:19.022523 43338 device_context.cc:352] device: 0, cuDNN Version: 7.4.
    W0104 17:59:23.193490 43338 fuse_all_reduce_op_pass.cc:78] Find all_reduce operators: 161. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 5.
    [Epoch 0, batch 0] loss: 0.12432, acc1: 0.00000, acc5: 0.06250
    [Epoch 0, batch 5] loss: 1.01921, acc1: 0.00000, acc5: 0.00000
    ...


完整2卡的日志信息也可在\ ``./log/``\ 目录下查看。

动态图使用方法
~~~~~~~~~

需要说明的是，动态图是天然支持Gradient Merge。即，只要不调用 ``clear_gradient`` 方法，动态图的梯度会一直累积。
动态图下使用Gradient Merge的代码片段如下：

.. code-block::

   for batch_id, data in enumerate(train_loader()):
       ... ...
       avg_loss.backward()
       if batch_id % k == 0:
           optimizer.minimize(avg_loss)
           model.clear_gradients()

