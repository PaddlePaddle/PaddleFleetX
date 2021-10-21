模型并行
------------------

简介
====

通常来讲，训练更大规模的网络模型可以在多种任务上取得更好的效果，如自然语言处理类
任务的准确率。然而，训练更大规模的网络模型会消耗更多的显存资源，甚至是超过单个设
备的显存容量，从而导致模型无法训练。模型并行通过将网络中的张量（Tensor）切分到不
同的设备，从而降低单个设备的显存消耗，使得超大规模模型训练成为可能。本文主要介绍
飞桨模型并行的基本原理和使用方法。

原理介绍
=======

自2017年提出以来， `Transformer <https://arxiv.org/abs/1706.03762>`__ 及其
变种模型成为自然语言类任务的常用模型，并于近年来被应用到图像视觉领域。
Transformer模型的基础结构是由Attention和MLP组成的Encoder和Decoder，以及
Embedding，如下图所示。其中Attention和MLP的底层实现均为矩阵乘法运算，而Embedding是一种
查找表实现。

.. image:: ../img/transformer_overview.png
  :width: 200
  :alt: transformer overview from the paper Megatron-LM
  :align: center

对于Embedding操作，可以将其理解为一种查找表操作。即，将输入看做索引，将Embedding参数
看做查找表，根据该索引查表得到相应的输出，如下图（a）所示。当采用模型并行时，
Embedding的参数被均匀切分到多个卡上。假设Embedding参数的维度为N*D，并采用K张卡执行模型
并行，那么模型并行模式下每张卡上的Embedding参数的维度为N//K*D。当参数的维度N不能被卡
数K整除时，最后一张卡的参数维度值为(N//K+N%K)*D。以下图（b）为例，Embedding参数的维度
为8*D，采用2张卡执行模型并行，那么每张卡上Embedding参数的维度为4*D。

为了便于说明，以下我们均假设Embedding的参数维度值D可以被模型并行的卡数D整除。此时，每张
卡上Embeeding参数的索引值为[0, N/K)，逻辑索引值为[k*N/K, (k+1)*N/K)，其中k表示卡序号，
0<=k<K。对于输入索引I，如果该索引在该卡表示的逻辑索引范围内，则返回该索引所表示的表项（索引
值为I-k*N/K；否则，返回值为全0的虚拟表项。随后，通过AllReduce操作获取所有输出表项的和，即
对应该Embeding操作的输出；整个查表过程如下图（b）所示。

.. image:: ../img/parallel_embedding.png
  :width: 800
  :alt: parallel embedding
  :align: center

对于矩阵乘操作，是按行或者列将矩阵切分K份。假设原始矩阵的维度为M*N，则按行切分后，各个
卡上的矩阵维度为M/K*N；若按列切分，则各个卡上矩阵的维度值为M*N/K。

下图给出按列切分矩阵乘法的示例图。其中，图（a）给出单卡上的矩阵乘法。图（b）给出模型并行
模式下的矩阵乘法，其中第二个矩阵按列切分到2张卡上；两张卡分别得到结果矩阵的一部分。最后，通过
AllGather通信操作汇聚最终的结果。

.. image:: ../img/col_parallel_matrix.png
  :width: 800
  :alt: column parallel matrix
  :align: center

下图给出按行切分矩阵乘法的示例图。其中，图（a）给出单卡上的矩阵乘法。图（b）给出模型并行
模式下的矩阵乘法，其中第二个矩阵按行切分到2张卡上；第一个矩阵需要按列切分，以满足矩阵乘法
的维度要求；两张卡分别得到结果矩阵的一部分。最后，通过
AllReduce通信操作按元素累加结果矩阵得到最终的结果。

.. image:: ../img/row_parallel_matrix.png
  :width: 800
  :alt: row parallel matrix
  :align: center

我们观察到，可以把上述按列切分矩阵乘法和按行切分矩阵乘法串联起来，从而省略掉一次AllGather通信
操作，如下图所示。同时，我们注意到Transformer的Attention和MLP组件中各种两次矩阵乘法操作。因此，我们
可以按照这种串联方式分别把Attention和MLP组件中的两次矩阵乘法串联起来，从而进一步优化性能。

.. image:: ../img/parallel_matrix.png
  :width: 800
  :alt: parallel matrix
  :align: center

我们观察到，在模型并行模式下，Transformer的Attention组件中存在两种类型的Dropout操作，如下图
所示。第一类是softmax算子后的Dropout算子；其输入是按列切分矩阵乘法的部分结果，我们称为局部
Dropout。直观理解，模型并行下，所有卡上的Dropout算子构成一个完整的Dropout算子，因此我们需要
确保不同卡上该类Dropout算子的丢弃位置是不同。第二类是图中g操作之后的Dropout操作，对于此类Dropout，其
输入均为完整且相同的输出，我们需要确保Dropout算子的输出也相同，即各个卡上该类Dropout算子选择
的丢弃位置是相同的。我们称此类Dropout为全局Dropout。我们通常通过设置种子来控制两类Dropout的输出。
具体地讲，对于局部Dropout，我们在不同的卡上为他们设置不同的种子，从而确保它们选择的丢弃位置是
不同的。而对于全局Dropout算子，我们在不同的卡上为它们设置相同的种子，从而确它们在不同卡上选择的
丢弃位置是相同的。

.. image:: ../img/global_local_dropout.png
  :width: 800
  :alt: dropout details from the paper Megatron-LM
  :align: center

我们需要注意一下几点：

- 模型并行下，需要确保模型并行组中各个卡读取相同的数据；
- 模型并行下，除了被切分的算子对应的输出外，其它所有算子的输出在各个卡上是一致的。

使用方法
=======

下面我们将分别介绍如何在静态图和动态图模式下使用飞桨模型并行。

静态图使用方法
~~~~~~~~~~~~~~~

在使用流水线并行的训练策略时，我们通过\ ``device_guard``\ 接口将不同的计算层放置在不同的设备上，如\ ``device_guard("gpu:0")``\ 。需要注意的是，当前流水线并行仅支持GPU设备。并且，模型中每个层都需要指定放置设备。

.. code-block:: python
   
   # device_guard 使用示例
   def build_network():
       with paddle.fluid.device_guard("gpu:0"):
           data = paddle.static.data(name='sequence', shape=[1], dtype='int64')
           data_loader = paddle.io.DataLoader.from_generator(
               feed_list=[data],
               capacity=64,
               use_double_buffer=True,
               iterable=False)
           emb = nn.embedding(input=data, size=[128, 64])
       with paddle.fluid.device_guard("gpu:1"):
           fc = nn.fc(emb, size=10)
           loss = paddle.mean(fc)
       return data_loader, loss

通过设定\ ``dist_strategy.pipeline`` 为True，将流水线并行的策略激活。

.. code-block:: python

   fleet.init(is_collective=True)
   dist_strategy = paddle.distributed.fleet.DistributedStrategy()
   dist_strategy.pipeline = True

进一步地，可以通过\ ``dist_strategy.pipeline_configs`` 配置流水线并行中mini-batch的切分粒度。假设mini-batch的大小为128，可以通过下述代码将mini-batch切为4份更小粒度的micro-batch，每个micro-batch的大小为32。需要注意地是，用户需要保证mini-batch大小是micro-batch大小的整数倍。

.. code-block:: python

   fleet.init(is_collective=True)
   dist_strategy = paddle.distributed.fleet.DistributedStrategy()
   strategy.pipeline_configs = {"accumulate_steps": 4,
                                "micro_batch_size": 32}


基于ResNet50网络的流水线并行代码：`example/resnet <https://github.com/PaddlePaddle/FleetX/tree/develop/examples/pipeline>`_。

使用下述命令行运行示例代码：

.. code-block:: python

   python -m paddle.distributed.launch \
          --gpus="0,1,2,3,4" \
          train_fleet_pipeline.py

控制台输出信息如下：

.. code-block:: python
   
   WARNING 2021-01-08 15:53:27,677 launch.py:314] Not found distinct arguments and compiled with cuda. Default use collective mode
   launch train in GPU mode
   INFO 2021-01-08 15:53:27,679 launch_utils.py:471] Local start 5 processes. First process distributed environment info (Only For Debug):
    +=======================================================================================+
    |                        Distributed Envs                      Value                    |
    +---------------------------------------------------------------------------------------+
    |                       PADDLE_TRAINER_ID                        0                      |
    |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:52033               |
    |                     PADDLE_TRAINERS_NUM                        5                      |
    |                PADDLE_TRAINER_ENDPOINTS  ... 0.1:12178,127.0.0.1:28915,127.0.0.1:32114|
    |                     FLAGS_selected_gpus                        0                      |
    +=======================================================================================+
    INFO 2021-01-08 15:53:27,679 launch_utils.py:475] details abouts PADDLE_TRAINER_ENDPOINTS can be found in log/endpoints.log.
    grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
    server not ready, wait 3 sec to retry...
    not ready endpoints:['127.0.0.1:40388', '127.0.0.1:12178', '127.0.0.1:28915', '127.0.0.1:32114']
    server not ready, wait 3 sec to retry...
    not ready endpoints:['127.0.0.1:12178']
    W0108 15:53:37.673019 103703 device_context.cc:342] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W0108 15:53:37.678391 103703 device_context.cc:352] device: 0, cuDNN Version: 7.6.

日志信息位于log目录下，log/workerlog.4日志文件的内容如下：

.. code-block:: python

   grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
   W0108 15:52:27.723405 103188 device_context.cc:342] Please NOTE: device: 4, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
   W0108 15:52:27.728278 103188 device_context.cc:352] device: 4, cuDNN Version: 7.6.
   I0108 15:52:32.665313 103188 gen_nccl_id_op_helper.cc:176] Server listening on: 127.0.0.1:32347 successful.
   W0108 15:52:36.874132 103188 operator.cc:1194] Device index is only supported under pipeline parallelism, so it will be ignored.
   grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
   W0108 15:53:31.393914 103723 device_context.cc:342] Please NOTE: device: 4, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
   W0108 15:53:31.398906 103723 device_context.cc:352] device: 4, cuDNN Version: 7.6.
   I0108 15:53:34.465754 103723 gen_nccl_id_op_helper.cc:176] Server listening on: 127.0.0.1:32114 successful.
   W0108 15:53:40.784844 103723 operator.cc:1194] Device index is only supported under pipeline parallelism, so it will be ignored.
   [Epoch 0, batch 5] loss: 0.37770, acc1: 0.03125, acc5: 0.03125
   [Epoch 0, batch 10] loss: 0.06200, acc1: 0.00000, acc5: 0.03125
   [Epoch 0, batch 15] loss: 0.26105, acc1: 0.00000, acc5: 0.00000
   [Epoch 0, batch 20] loss: 0.00000, acc1: 0.00000, acc5: 0.00000
   [Epoch 0, batch 25] loss: 0.37330, acc1: 0.00000, acc5: 0.06250
   [Epoch 0, batch 30] loss: 0.00000, acc1: 0.00000, acc5: 0.00000
   [Epoch 0, batch 35] loss: 0.07487, acc1: 0.00000, acc5: 0.00000
   [Epoch 0, batch 40] loss: 0.12932, acc1: 0.03125, acc5: 0.06250
   [Epoch 0, batch 45] loss: 0.19604, acc1: 0.00000, acc5: 0.03125
   [Epoch 0, batch 50] loss: 0.07977, acc1: 0.00000, acc5: 0.00000
   [Epoch 0, batch 55] loss: 0.00000, acc1: 0.00000, acc5: 0.00000
   [Epoch 0, batch 60] loss: 0.13464, acc1: 0.00000, acc5: 0.06250
   [Epoch 0, batch 65] loss: 0.13940, acc1: 0.00000, acc5: 0.03125
   [Epoch 0, batch 70] loss: 0.00000, acc1: 0.00000, acc5: 0.00000
   [Epoch 0, batch 75] loss: 0.00000, acc1: 0.00000, acc5: 0.00000

动态图使用方法
~~~~~~~~~~~~~~~
