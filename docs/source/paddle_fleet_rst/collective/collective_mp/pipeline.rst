流水线并行
------------------

简介
====

通常来讲，训练更大规模的网络模型可以在多种任务上取得更好的效果，如提升图像分类任务的准确率。然而，训练更大规模的网络模型会消耗更多的显存资源，甚至是超过单个设备的显存容量，从而导致模型无法训练。流水线并行通过将网络模型不同的层放置到不同的设备，从而降低单个设备的显存消耗，使得超大规模模型训练成为可能。本文主要介绍飞桨流水线并行的基本原理和使用方法。

原理介绍
=======

.. image:: ../img/pipeline-1.png
  :width: 400
  :alt: pipeline
  :align: center

与数据并行不同，流水线并行将模型的不同层放置到不同的计算设备，降低单个计算设备的显存消耗，从而实现超大规模模型训练。以上图为例，示例模型包含四个模型层。该模型被切分为三个部分，并分别放置到三个不同的计算设备。即，第1层放置到设备0，第2层和第三3层放置到设备1，第4层放置到设备2。相邻设备间通过通信链路传输数据。具体地讲，前向计算过程中，输入数据首先在设备0上通过第1层的计算得到中间结果，并将中间结果传输到设备1，然后在设备1上计算得到第2层和第3层的输出，并将模型第3层的输出结果传输到设备2，在设备2上经由最后一层的计算得到前向计算结果。反向传播过程类似。最后，各个设备上的网络层会使用反向传播过程计算得到的梯度更新参数。由于各个设备间传输的仅是相邻设备间的输出张量，而不是梯度信息，因此通信量较小。

下图给出流水线并行的时序图。最简配置流水线并行模型下，任意时刻只有单个计算设备处于计算状态，其它计算设备则处于空闲状态，因此设备利用率和计算效率较差。

.. image:: ../img/pipeline-2.png
  :width: 600
  :alt: pipeline_timeline1
  :align: center

为了优化流水线并行中设备的计算效率，可以进一步将mini-batch切分成若干更小粒度的micro-batch，以提升流水线并行的并发度，进而达到提升设备利用率和计算效率的目的。如下图所示，一个mini-batch被切分为4个micro-batch；前向阶段，每个设备依次计算单个micro-batch的结果；从而增加了设备间的并发度，降低了流水线并行bubble空间比例，提高了计算效率。

.. image:: ../img/pipeline-3.png
  :width: 600
  :alt: pipeline_timeline2
  :align: center

功能效果
=======

使用流水线并行，可以实现超大规模模型训练。例如，使用多个计算设备，可以实现单个计算设备显存无法容纳的模型训练。

使用方法
=======

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
   strategy.pipeline_configs = {"micro_batch": 4}


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

注意事项
=======
由于流水线并行将模型的层放置到不同的计算设备，因此在fetch信息时，只有所fetch的数据所在设备进程对应的日志信息中输出数据信息，其它设备进程对应的日志输出None。以上面的示例说明，由于获取的损失值和精度值只在最后一个设备上，因此只有log/workerlog.4日志文件中会输出对应的数据，其它日志文件不会输出对应的数据。
