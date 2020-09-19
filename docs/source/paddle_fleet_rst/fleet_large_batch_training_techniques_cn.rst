使用超大Batch进行训练
=====================

简介 + strategy列表
-------------------

为了追求模型的性能不断提升，人们对更大规模的数据集、更深的网络层、更庞大的参数规模趋之若鹜。但是随之而来的就是给模型训练带来了巨大的压力，因此分布式技术及定制化AI芯片应运而生。但在分布式训练中，经常会遇到显存或者内存不足的情况，通常是以下几点原因导致的：

-  输入的数据过大，例如视频类训练数据。
-  深度模型的参数过多或过大，所需的存储空间超出了内存/显存的大小。
-  AI芯片的内存有限。

为了能正常完成训练，我们通常只能使用较小的Batch
Size以降低模型训练中的所需要的存储空间，这将导致很多模型无法通过提高训练时的Batch
Size来提高模型的精度。为了解决这个问题，Fleet中提供了两种策略，使得模型可以使用超大Batch的方式完成训练：

-  **Forward Recomputation Backpropagation（FRB）：**
   通过清除正向计算过程中的中间计算结果，来降低训练过程中使用的存储空间，从而确保硬件有足够的内存做更大Batch
   Size的训练。
-  **Gradient Merge：**
   在训练过程中，将连续多个Batch数据训练得到的梯度合并更新模型参数的策略。在该训练策略下，虽然从形式上看依然是小Batch规模的数据在训练，但是效果上可以达到多个小Batch数据合并成大Batch后训练的效果。

原理
----

Forward Recomputation Backpropagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们知道，深度学习网络的一次训练迭代包含三个步骤：

-  **前向计算：** 运行前向算子(Operator) 来计算中间隐层(Variable)的值 。
-  **反向计算：** 运行反向算子来计算参数(Parameter)的梯度。
-  **优化：** 应用优化算法以更新参数值 。

在前向计算过程中，前向算子会计算出大量的中间结果，由于这些中间结果是训练数据和算子计算得到的，所以训练数据的Batch
Size越大，中间结果占用的内存也就越大。飞桨核心框架会使用
Variable来存储这些隐层的中间结果。当模型层数加深时，其中间结果的数量可达成千上万个，
占据大量的内存。虽然飞桨核心框架的显存回收机制会及时清除无用的中间结果，以节省存储。
但是有些中间结果是反向计算过程中算子的输入，这些中间结果必须存储在内存中，直到相应的反向算子计算完毕。

对于大小固定的内存来说，如果用户希望使用大Batch
Size的数据进行训练，则将导致单个中间结果占用内存增大，那么就需要减少中间结果的存储数量，FRB就是基于这种思想设计的。

FRB是将深度学习网络切分为k个部分（segments）。对每个segment而言：前向计算时，除了小部分必须存储在内存中的Variable外，其他中间结果都将被删除；在反向计算中，首先重新计算一遍前向算子，以获得中间结果，再运行反向算子。简而言之，FRB和普通的网络迭代相比，多计算了一遍前向算子。

我们把切分网络的变量叫做checkpoints。
那么问题来了，如何选择checkpoints呢？自从FRB方法提出以来，大量学者在研究这一关键问题。
我们知道深度学习网络通常是由一个个模块串联得到的，比如ResNet-50由16个block串联而成，
Bert-Large由24个transformer串联而成，以两个子模块中间的变量作为切分点就是一个很好的选择。
对于非串联的网络（比如含有大量shortcut结构的网络），FRB也支持对其做切分，
只是可能多耗费一点内存（用于存储shortcut的Variable）。

Gradient Merge
~~~~~~~~~~~~~~

与FRB相比，Gradient
Merge并没有像FRB那样对内存的使用做出大刀阔斧般的改动，只是在训练流程上做了一些微调，达到模拟出大Batch
Size训练效果的目的。具体来说，就是使用若干原有大小的Batch数据进行训练，即通过“前向+反向”
网络计算得到梯度。其间会有一部分显存/内存用于存放梯度，然后对每个Batch计算出的梯度进行叠加，在计算完所有Batch后，使用累加的梯度对模型进行参数更新，从而达到使用大Batch数据训练的效果。

GradientMerge
策略在使用方面也很简单，用户只需要定义将多少Batch的数据计算出的梯度叠加更新模型参数，便可以实现大Batch训练的目的。

操作实践
--------

该章节中我们将基于BERT模型的实用样例，分别对这两个增大Batch的策略进行讲解。从整体来看，训练脚本的编写主要分为4个部分：

-  添加训练脚本运行所必须的依赖包。
-  定义分布式模式并初始化。
-  加载模型及数据。
-  定义训练策略和优化器，在这一步我们可以选择使用FRB或者Gradient
   Merge策略来增大BatchSize。

下面我们来分别介绍FRB和Gradient
Merge两种策略所对应脚本的编写方法。在开始之前，我们需要准备训练数据集（train_data.tar.gz）及词表（vocab.txt）。

.. code:: sh

   wget --no-check-certificate https://fleet.bj.bcebos.com/Bertdata/train_data.tar.gz
   tar -xf train_data.tar.gz
   wget --no-check-certificate https://fleet.bj.bcebos.com/Bertdata/vocab.txt

.. _forward-recomputation-backpropagation-1:

Forward Recomputation Backpropagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

添加依赖
^^^^^^^^

首先我们需要添加训练中所用到的python模块，\ ``fleetx``
可以用于加载我们为用户封装的接口如：加载模型及数据，模型训练等。\ ``paddle.distributed.fleet``
中定义了丰富的分布式策略供用户使用。

.. code:: python

   import fleetx as X
   import paddle.fluid as fluid
   import paddle.distributed.fleet as fleet
   import paddle.distributed.fleet.base.role_maker as role_maker

定义分布式模式并初始化
^^^^^^^^^^^^^^^^^^^^^^

通过\ ``X.parse_train_configs()``\ 接口，用户可以定义训练相关的参数，如：学习率、衰减率等。同时通过\ ``fleet.init()``\ 接口定义了分布式模型，下面代码中的\ ``is_collective=True``\ 表示采用集合通信的GPU分布式模式训练模型。

.. code:: python

   configs = X.parse_train_configs()
   role = role_maker.PaddleCloudRoleMaker(is_collective=True)
   fleet.init(role)

加载模型及数据
^^^^^^^^^^^^^^

用户可以通过\ ``X.applications``\ 接口加载我们预先定义好的模型，如：Resnet50、VGG16、BERT等。并使用定制化的data_loader加载模型，同时可以定义训练中使用的batch_size等参数。下面的例子中，我们使用了recompute对Bert_large模型所支持的最大Batch
Size（53）来进行训练。

.. code:: python

   model = X.applications.Bert_large()

   data_loader = model.load_digital_dataset_from_file(
       data_dir='./train_data',
       vocab_path='./vocab.txt',
       max_seq_len=512,
       batch_size=53,
   )

定义Recompute Strategy 及 Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

接下来我们就可以定义分布式训练中所应用到的策略了。下面的例子中，为了使用Recompute策略，我们将\ ``dist_strategy.recompute``\ 设置为True
并设置我们事先定义好的checkpoints。

接下来用户需要定义训练中更新模型所用到的优化器，并使用\ ``fleet.distributed_optimizer``\ 接口将优化器转换为分布式模式。

最后运行\ ``optimizer.minimize(model.loss)``
将反向计算的算子插入训练网络，我们就可以开始训练了。

.. code:: python

   dist_strategy = fleet.DistributedStrategy()
   # 使用Recompute，并设置checkpoints
   dist_strategy.recompute = True
   dist_strategy.recompute_configs = {"checkpoints": model.checkpoints}

   optimizer = fluid.optimizer.Adam(learning_rate=configs.lr)
   optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
   optimizer.minimize(model.loss)

开始训练
^^^^^^^^

在 FleetX 中，我们为用户提供了\ ``X.MultiGPUTrainer``
接口，用于GPU分布式训练。其中\ ``model`` 及 ``data_loader``
分别为第二步中加载的模型及数据。\ ``start_step``
表示开始打印训练log的步数，若用户想复现我们的模型训练速度数据建议设置成10或者更大的数；若用户想查看模型的收敛情况，则可设置成0。

.. code:: python

   trainer = X.MultiGPUTrainer()
   trainer.fit(model, data_loader, start_step=10)

运行训练脚本
~~~~~~~~~~~~

完成脚本的编写后我们就可以使用以下命令训练分布式模型：

.. code:: sh

   fleetrun --gpus 0,1,2,3,4,5,6,7 bert_recompute.py

效果测试
^^^^^^^^

我们在BERT模型上对recompute的效果进行了测试，使用Recompute后Batch
size可以扩大至3倍。与混合精度一起使用时，Batch_size可以进一步扩大。

-  **Bert_large**:

========== ============ ============= ===========================
Model      Baseline     Recompute     Recompute + mixed precision
========== ============ ============= ===========================
Batch size 14           53            87
speed      18.2 sents/s 12.88 sents/s 19.14 sents/s
========== ============ ============= ===========================

.. _gradient-merge-1:

Gradient Merge
~~~~~~~~~~~~~~

下面，我们介绍如何使用 Gradient Merge 来扩大BERT模型分布式训练中的 Batch
Size（假设脚本名称为bert_gradient_merge.py）：

与 Forward Recompute Backpropagation
相同，我们首先要添加依赖，定义分布式模式并加载模型及数据。

.. _添加依赖-1:

添加依赖
^^^^^^^^

.. code:: python

   import fleetx as X
   import paddle.fluid
   import paddle.distributed.fleet as fleet
   import paddle.distributed.fleet.base.role_maker as role_maker

.. _定义分布式模式并初始化-1:

定义分布式模式并初始化
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   configs = X.parse_train_configs()
   role = role_maker.PaddleCloudRoleMaker(is_collective=True)
   fleet.init(role)

.. _加载模型及数据-1:

加载模型及数据
^^^^^^^^^^^^^^

.. code:: python

   model = X.applications.Bert_large()

   data_loader = model.load_digital_dataset_from_file(
       data_dir='./train_data',
       vocab_path='./vocab.txt',
       max_seq_len=512,
       batch_size=13,
   )

定义Gradient Merge Strategy 及 Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在上面的代码中，我们定义了Batch
Size为13，在这一步中，我们将设置使用4个Batch
Size来模拟一个大Batch的训练，从而达到了Batch size为52的训练效果。

在\ ``gradient_merge_configs``\ 中，avg选项用于控制梯度累计的形式：当被设置为
True
时，会对每次的梯度求和并做平均；反之将直接对梯度求和，并对参数进行更新。

.. code:: python

   dist_strategy = fleet.DistributedStrategy()
   # 使用Gradient merge策略并设置相关参数
   dist_strategy.gradient_merge = True
   dist_strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}
   optimizer = fluid.optimizer.Adam(learning_rate=configs.lr)
   optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
   optimizer.minimize(model.loss)

.. _开始训练-1:

开始训练
^^^^^^^^

Gradient Merge 的训练代码与 Recompute
策略相同，用户使用两行代码即可开始训练：

.. code:: python

   trainer = X.MultiGPUTrainer()
   trainer.fit(model, data_loader, start_step=10)

.. _运行训练脚本-1:

运行训练脚本
^^^^^^^^^^^^

.. code:: sh

   fleetrun --gpus 0,1,2,3,4,5,6,7 bert_gradient_merge.py
