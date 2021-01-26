飞桨大规模分类库使用介绍
------------------------

简介
====

图像分类技术日趋成熟，ResNet网络在ImageNet数据集上的top5准确率已超过96%。然而，如何高效地完成百万类别甚至是更大规模的分类任务，则是一个极具挑战性的课题。

从多分类神经网络的实现角度分析，其最后一层通常是由全连接层和Softmax构成的组合层，全连接层输出结点数挂钩分类任务的类别数，所以对应的参数量随分类类别数的增长而线性增长。因此，当类别数非常大时，神经网络训练过程占用的显存空间也会很大，甚至是超出单张GPU卡的显存容量，导致神经网络模型无法训练。

以新闻推荐系统为例，假设要对百万类细分类别的新闻条目进行分类，那么仅存储全连接层参数就需要约2GB的显存空间（这里假设神经网络最后一层隐层的输出结点的维度为512，并假设以32比特浮点数表示数据，见下式）。再考虑神经网络训练过程中生成的数量庞多的中间变量，那么训练过程中需要的存储总量往往会超出单张GPU卡的显存容量。

$$全连接层参数显存消耗=\\frac{512*10^6*4B}{1024^3}\\approx2GB$$

原理介绍
========

该如何解决这个问题呢？常用的做法是“拆分”。考虑到全连接层的线性可分性，可以将全连接层参数切分到多张GPU卡，采用模型并行方案，减少每张GPU卡的参数存储量。

以下图为例，全连接层参数按行切分到不同的GPU卡上。每次训练迭代过程中，各张GPU卡分别以各自的训练数据计算隐层的输出特征(feature)，并通过集合通信操作AllGather得到汇聚后的特征。接着，各张GPU卡以汇聚后的特征和部分全连接层参数计算部分logit值(partial logit)，并基于此计算神经网络的损失值。详细推导过程请参阅附录。

.. image:: ../img/plsc_overview.png
  :width: 400
  :alt: plsc
  :align: center

这个方案可以有效解决全连接层参数量随分类类别数线性增长导致的显存空间不足的问题。然而，为了实现这一方案，开发者需要基于现有的深度学习平台设计和实现上例描述的所有操作，包括全连接层参数的切分和集合通信等，动辄需要数百行实现代码，大大增加了开发者的负担。飞桨大规模分类库(PLSC: PaddlePaddle Large Scale Classification)，为用户提供了大规模分类任务从训练到部署的全流程解决方案。只需数行代码，即可实现千万类别分类的神经网络。并且，通过PLSC库提供的serving功能用户可以快速部署模型，提供一站式服务。

功能效果
========

PLSC库在多个数据集上可以取得SOTA的训练精度，下表列出PLSC库分别使用MS1M-ArcFace和CASIA数据集作为训练数据，在不同验证数据集上取得的精度。

.. list-table::
   :header-rows: 1

   * - 模型
     - 训练集
     - lfw
     - agendb_30
     - cfp_ff
     - cfp_fp
     - MegaFace (Id/Ver)
   * - ResNet50
     - MS1M-ArcFace
     - 0.99817
     - 0.99827
     - 0.99857
     - 0.96314
     - 0.980/0.993
   * - ResNet50
     - CASIA
     - 0.98950
     - 0.90950
     - 0.99057
     - 0.91500
     - N/A


备注：上述模型训练使用的loss_type为'dist_arcface'。更多关于ArcFace的内容请参考: `ArcFace: Additive Angular Margin Loss for Deep Face Recognition <https://arxiv.org/abs/1801.07698>`_。

PLSC支持多机分布式训练。一方面，通过多机分布式训练可以将全连接层参数切分到更多的GPU卡，从而支持千万类别分类，并且飞桨大规模分类库理论上支持的分类类别数随着使用的GPU卡数的增加而增加。例如，单机8张V100 GPU配置下支持的最大分类类别数相比不使用PLSC扩大2.52倍。另一方面，使用多机分布式训练可以有效提升训练速度。

下图给出使用不同数量的节点时的训练速度（吞吐）。实验中使用的训练数据集为MS1M-ArcFace，分类类别数为85742，每个节点配备8张NVIDIA V100 GPUs，backbone模型为ResNet50。如图所示，使用飞桨大规模分类库可以取得近似线性的加速比。

.. image:: ../img/plsc_performance.png
   :target: ./plsc_performance.png
   :alt: performance
   :align: center

使用方法
========

安装PLSC
^^^^^^^^

执行下面的命令安装PLSC。

.. code-block:: shell

   pip install plsc

准备模型训练配置代码，保存为train.py文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用PLSC组建分类神经网络主要包括下面三个步骤：


#.
   从plsc包导入Entry类，Entry类封装PLSC所有API的接口类；

#.
   实例化Entry类的对象；

#.
   调用Entry类的train方法，开始训练过程。

默认情况下，该训练脚本使用的loss值计算方法为'dist_arcface'，即将全连接层参数切分到多张GPU卡的模型并行方案，需要使用两张或以上的GPU卡。默认地，基础模型为ResNet50模型，关于如何自定义模型请参考 `文档 <https://github.com/PaddlePaddle/PLSC/blob/master/docs/source/md/advanced.md>`_。

.. code-block:: python

   from plsc import Entry
   if __name__ == "main":
           ins = Entry()
           ins.set_class_num(1000000) #设置分类类别数
           ins.train()

启动训练任务
^^^^^^^^^^^^

可以使用下面的命令行启动训练任务，其中selected_gpus参数用于指定训练中使用的GPU卡。

.. code-block:: shell

   python -m paddle.distributed.launch \
               --selected_gpus=0,1,2,3,4,5,6,7 \
               train.py


更多PLSC使用文档，请参阅: `PLSC Repo <https://github.com/PaddlePaddle/PLSC>`_。

附录
====

全连接层操作在数学上等价于输入X和参数W的矩阵乘: :math:`XW`。参数W可以按列切分为N个部分 :math:`[W_{0}, W_{1}, ..., W_{N-1}]`，并分别放置到N张卡上。

$$XW = X[W_{0}, W_{1}, ..., W_{N-1}] = [XW_{0}, XW_{1}, ..., XW_{N-1}]$$

因此，在第i张卡上，只需要计算部分结果 :math:`XW_{i}`。然后，通过集合通信操作获取全局结果 :math:`XW`。
