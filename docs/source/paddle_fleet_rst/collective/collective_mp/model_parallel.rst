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

静态图中，我们提供了 `paddle.distributed.split <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/split_cn.html#split>`_ 实现
Embedding和矩阵乘法算子的切分。我们需要对该API的 ``gather_out`` 参数做一些特殊说明：对于Embedding切分操作，该参数始终设置
为True。对于矩阵切分操作，如果该参数设置为True，则会在算子操作后使用通信操作获取最终结果。参照上文，对于连续的两个切分的矩阵
乘法操作，我们通常对第一个矩阵乘法采用按列切分方法，对第二个矩阵乘法采用按行切分方法；并且，对于按列切分的矩阵乘法，我们
将 ``gather_out`` 参数设置为False，从而省略掉一次通信操作。

下面的例子给出在两张卡上实现Embedding算子模型并行的示例。

.. code-block:: python
   
   emb_out = padle.distributed.split(
    in,
    (8, 8),
    operation="embedding",
    num_partitions=2)   

此外，我们还需要配置Fleet的选项，以使用模型并行功能。

.. code-block:: python

   fleet.init(is_collective=True)
   dist_strategy = paddle.distributed.fleet.DistributedStrategy()
   dist_strategy.tensor_parallel = True
   strategy.tensor_parallel_configs = {"tensor_parallel_degree": 4}

其中， ``tensor_parallel_degree`` 指定模型并行的并行度。

如上文所述，对于Transformer模型，存在两种类型的Dropout：全局Dropout和局部Dropout；对于
全局Dropout，需要在模型并行的所有卡上设置相同的种子，对于局部Dropout，则需要设置不同的
种子。我们通过如下代码分别设置全局和局部种子：

.. code-block:: python

   mp_local_seed = basic_seed + mp_rank * 11
   mp_global_seed = basic_seed
   paddle.framework.random.set_random_seed_generator('mp_local_seed', mp_local_seed)
   paddle.framework.random.set_random_seed_generator('mp_global_seed', mp_global_seed)

上例只是一种示例实现，用户可以根据自己的需要实现不同的种子设置方式，但需要确保同一模型并行
组内，全局Dropout的种子是一致的，而局部Dropout的种子是不同的。

在使用 ``dropout`` 接口时，我们还需要根据其类型设置其种子参数，如下例所示：

.. code-block:: python

   # For local dropout
   weights = dropout(
                     weights,
                     p=dropout_rate,
                     rng_name='mp_local_seed',
                     training=True,
                     mode='upscale_in_train')

   # For global dropout
   weights = dropout(
                     weights,
                     p=dropout_rate,
                     rng_name='mp_global_seed',
                     training=True,
                     mode='upscale_in_train')

动态图使用方法
~~~~~~~~~~~~~~~
