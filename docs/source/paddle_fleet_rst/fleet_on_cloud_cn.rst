优化低配网络的分布式GPU训练
===========================

在网络带宽较低的训练场景（如：
共有云上训练，联邦训练）中，梯度同步在低带宽网络下的延迟成为训练速度的主要瓶颈。
Fleet 实现了： ``Deep Gradient Compression`` 和 ``local SGD``
两种训练策略来针对性解决这一问题。

DGC 优化低配网络的分布式GPU训练
-------------------------------

DGC 简介
~~~~~~~~

大规模分布式训练需要较高的网络带宽以便进行梯度的聚合更新，这限制了多节点训练时的可扩展性同时也需要昂贵的高带宽设备。在低带宽云网络等环境下进行分布式训练会变得更加糟糕。
`Deep Gradient Compression <https://arxiv.org/abs/1712.01887>`__
发现：分布式SGD中有99.9%的梯度交换都是冗余的，可以使用深度梯度压缩选择重要梯度进行通信来减少通信量，降低对通信带宽的依赖。FleetX
实现了DGC的稀疏通信方式，可有效在低配网络下进行GPU分布式训练。FleetX
实现了 DGC 论文中的 ``预热训练 (warming up training)``,
``动量修正 (Momentum Correction)``,
``局部梯度修剪 (local gradient clipping)``,
``动量因子掩藏 (Momentum factor masking)`` 等策略， 和
``正则化项修正 (Weight Decay Correction)``
避免稀疏梯度通信训练带来的最终模型精度损失。

下面将介绍 DGC 稀疏通信方式的适用场景及、基本原理，FleetX 中 DGC
的效果和使用方法。

适用场景
^^^^^^^^

DGC稀疏通信在低带宽通信瓶颈时会有较大的性能提升，但\ **在单机多卡及RDMA网络通信并非瓶颈情况下**\ ，并不会带来性能上的提升。同时由于AllGather的通信量会随卡数的增多而增大，所以DGC的多机训练规模也不宜过大。故DGC适用于低配网络，同时节点规模不宜过大，如>128张卡。在云网络或高带宽网络设备昂贵时，DGC可有效降低训练成本。

FleetX 效果
~~~~~~~~~~~

-  模型：FasterRCNN
-  硬件： P40两机分布式，每台机器一卡，TCP网络测试。
-  取300-700步耗时/400step。
-  精度无损。

==== ========================== ============================= =====
带宽 训练耗时-Momentum(step /s) 训练耗时-DGCMomentum(step /s) 比率
==== ========================== ============================= =====
100G 0.3725                     0.375                         0.993
10G  0.55                       0.375                         1.467
1G   2.45                       0.375                         6.533
==== ========================== ============================= =====

DGC 原理
~~~~~~~~

梯度稀疏
^^^^^^^^

DGC的基本思路是通过只传送重要梯度，即只发送大于给定阈值的梯度来减少通信带宽的使用。为避免信息的丢失，DGC会将剩余梯度在局部累加起来，最终这些梯度会累加大到足以传输。
换个角度，从理论依据上来看，局部梯度累加等同于随时间推移增加batch
size，（DGC相当于每一个梯度有自己的batch size）。设定 :math:`F(w)`
为需要优化的loss函数，则有着N个训练节点的同步分布式SGD更新公式如下

.. math::

   F(w)=\\frac{1}{\|\\chi\|}\\sum\_{x\\in\\chi}f(x, w), \\qquad w\_{t+1}=w\_{t}-\\eta\\frac{1}{N b}\\sum\_{k=1}^{N}\\sum\_{x\\in\\mathcal{B}\_{k,t}}\\nabla f\\left(x, w\_{t}\\right) \\tag{1}

其中\ :math:`\chi`\ 是训练集，\ :math:`w`\ 是网络权值，\ :math:`f(x, w)`\ 是每个样本\ :math:`x \in \chi`\ 的loss，\ :math:`\eta`\ 是学习率，N是训练节点个数，\ :math:`\mathcal{B}\_{k, t}`\ 代表第\ :math:`k`\ 个节点在第\ :math:`t`\ 个迭代时的minibatch，大小为b。
考虑权重的第i个值，在T次迭代后，可获得

.. math::

   w\_{t+T}^{(i)}=w\_{t}^{(i)}-\\eta T \\cdot \\frac{1}{N b T} \\sum\_{k=1}^{N}\\left(\\sum\_{\\tau=0}^{T-1} \\sum\_{x \\in \\mathcal{B}\_{k, t+\\tau}} \\nabla^{(i)} f\\left(x, w\_{t+\\tau}\\right)\\right)  \\tag{2}

等式2表明局部梯度累加可以被认为batch
size从\ :math:`Nb`\ 增大为\ :math:`NbT`\ ，其中T是\ :math:`w^{(i)}`\ 两次更新的稀疏通信间隔。

预热调参
^^^^^^^^

对于正常的训练，使用DGC一般需进行预热训练，否则可能会有精度损失。如下图是ResNet50模型Imagenet数据集的训练结果，未进行预热训练的DGC最终损失了约0.3%的精度。

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

预热训练调参可参照论文的设置。论文中使用了 75%, 93.75%, 98.4375%, 99.6%,
99.9%
稀疏度逐渐提升的策略。由于paddle稀疏梯度聚合通信使用了AllGather，通信量会随卡数增加而增长，所以在卡数较多时不推荐较低稀疏度的预热训练。如75%稀疏度时每张卡会选择25%的梯度进行通信，卡数为32时通信量是正常dense通信的32*(1-0.75)=8倍，所以前几个epoch使用正常的dense通信为佳。可参照如下设置参数：

.. code:: python

   # 1. 以1252个step为一个epoch，前2个epochs使用正常dense通信，后3个epochs逐步提升稀疏度为99.9%
   strategy.dgc_configs = {
       "rampup_begin_step": 1252*2,
       "rampup_step": 1252*3,
       "sparsity": [0.984375, 0.996, 0.999]
   }
   # 2. 前面4个epochs都使用dense通信，之后默认0.999稀疏度运行
   strategy.dgc_configs = {
       "rampup_begin_step": 1252*4,
       "rampup_step": 1,
       "sparsity": [0.999]
   }

对于Fine-tuning训练，现测试可无需预热训练，从第0个epoch直接使用DGC即可。

.. code:: python

   # 从第0步开始DGC稀疏通信
   strategy.dgc_configs = {
       "rampup_begin_step": 0,
       "rampup_step": 1,
       "sparsity": [0.999]
   }

局部梯度累加改进
^^^^^^^^^^^^^^^^

正常情况，稀疏更新会严重影响收敛性。DGC中采用动量修正(Momentum
Correction)和局部梯度裁减(Local Gradient Clipping)来解决这个问题。

动量修正
''''''''

有着N个节点分布式训练中vanilla momentum SGD公式，

.. math::

   u\_{t}=m u\_{t-1}+\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right), \\quad w\_{t+1}=w\_{t}-\\eta u\_{t}  \\tag{3}

其中\ :math:`m`\ 是动量因子，\ :math:`N`\ 是节点数，\ :math:`\\nabla\_{k, t}=\\frac{1}{N b} \\sum\_{x \\in \\mathcal{B}\_{k, t}} \\nabla f\\left(x, w\_{t}\\right)`\ 。
考虑第i个权重\ :math:`w^{(i)}`\ ，在T次迭代后，权重更新公式如下，

.. math::

   w\_{t+T}^{(i)}=w\_{t}^{(i)}-\\eta\\left[\\cdots+\\left(\\sum\_{\\tau=0}^{T-2} m^{\\tau}\\right) \\nabla\_{k, t+1}^{(i)}+\\left(\\sum\_{\\tau=0}^{T-1} m^{\\tau}\\right) \\nabla\_{k, t}^{(i)}\\right]  \\tag{4}

如果直接应用动量SGD到稀疏梯度更新中，则有公式，

.. math::

   v\_{k, t}=v\_{k, t-1}+\\nabla\_{k, t}, \\quad u\_{t}=m u\_{t-1}+\\sum\_{k=1}^{N} \\operatorname{sparse}\\left(v\_{k, t}\\right), \\quad w\_{t+1}=w\_{t}-\\eta u\_{t} \\tag{5}

其中\ :math:`v\_k`\ 是训练节点k上的局部梯度累加项，一旦\ :math:`v\_k`\ 大于某一阈值，则会在第二项中压缩梯度进行动量更新，并使用sparse()函数获得mask清空大于阈值的梯度。
:math:`w^{(i)}`\ 在T次稀疏更新后的权重为,

.. math::

   w\_{t+T}^{(i)}=w\_{t}^{(i)}-\\eta\\left(\\cdots+\\nabla\_{k, t+1}^{(i)}+\\nabla\_{k, t}^{(i)}\\right) \\tag{6}

相比传统动量SGD，方程6缺失了累积衰减因子\ :math:`\sum\_{\tau=0}^{T-1} m^{\tau}`\ ，会导致收敛精度的损失。如下图(a)，正常梯度更新从A点到B点，但是方程6则从A点到C点。当稀疏度很高时，会显著降低模型性能，所以需要在方程5基础上对梯度进行修正。

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

若将方程3中速度项\ :math:`u\_t`\ 当作“梯度”，则方程3第二项可认为是在”梯度“\ :math:`u\_t`\ 上应用传统SGD，前面已经证明了局部梯度累加在传统SGD上是有效的。因此，可以使用方程3局部累加速度项\ :math:`u\_t`\ 而非累加真实的梯度\ :math:`\nabla\_{k, t}`\ 来修正方程5，

.. math::

   u\_{k, t}=m u\_{k, t-1}+\\nabla\_{k, t}, \\quad v\_{k, t}=v\_{k, t-1}+u\_{k, t}, \\quad w\_{t+1}=w\_{t}-\\eta \\sum\_{k=1}^{N} \\operatorname{sparse}\\left(v\_{k, t}\\right)  \\tag{7}

修正后，如上图(b)，方程可正常从A点到B点。除了传统动量方程修正，论文还给出了Nesterov动量SGD的修正方程。

局部梯度修剪
''''''''''''

梯度修剪是防止梯度爆炸的常用方法。这方法由Pascanu等人在2013年提出，当梯度的l2-norms和大于给定阈值时，就对梯度rescale。正常梯度修剪在梯度聚合后使用，而DGC因为每个节点独立的进行局部梯度累加，所以DGC在使用\ :math:`G\_t`\ 累加前对其进行局部梯度修剪。阈值缩放为原来的\ :math:`N^{-1/2}`

.. math::

   thr\_{G^{k}}=N^{-1 / 2} \\cdot thr\_{G}  \\tag{8}

#### 克服迟滞效应
因为推迟了较小梯度更新权重的时间，所以会有权重陈旧性问题。稀疏度为99.9%时大部分参数需600到1000步更新一次。迟滞效应会减缓收敛并降低模型精度。DGC中采用动量因子掩藏和预热训练来解决这问题。

动量因子掩藏
''''''''''''

DGC中使用下面方程来掩藏动量因子减缓陈旧性问题。

.. math::

   Mask \\leftarrow\\left|v\_{k, t}\\right|>t h r, \\quad v\_{k, t} \\leftarrow v\_{k, t} \\odot \\neg Mask, \\quad u\_{k, t} \\leftarrow u\_{k, t} \\odot \\neg Mask \\tag{9}

此掩码可以停止延迟梯度产生的动量，防止陈旧梯度把权重引入错误的方向。

正则化(Weight Decay)项修正
^^^^^^^^^^^^^^^^^^^^^^^^^^

Paddle框架以Weight
Decay的形式实现正则化。以L2Decay为例，公式(3)中传统momentum添加weight
decay后公式为

.. math::

   G\_{t}=\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right)+\\lambda w\_{t}, \\quad  u\_{t}=m u\_{t-1}+G\_{t}, \\quad w\_{t+1}=w\_{t}-\\eta u\_{t} \\tag{10}

其中\ :math:`\lambda`\ 为Weight
Decay系数，\ :math:`G\_{t}`\ 为添加L2Decay项之后的聚合梯度。由于在公式7中进行了局部动量修正，所以按照相同思路在局部梯度上运用修正的Weight
Decay项。如下公式在局部梯度上添加局部Weight Decay项即可。

.. math::

   \\nabla\_{k, t}=\\nabla\_{k, t}+\\frac{\\lambda}{N} w\_{t} \\tag{11}

在模型实际训练中，通常会设置weight
decay的系数\ :math:`\lambda=10^{-4}`\ ，在卡数较多如4机32卡的情况下局部weight
decay系数为\ :math:`\frac{\lambda}{N}=\frac{10^{-4}}{32}=3.125\*10^{-6}`\ ，在数值精度上偏低，测试训练时会损失一定精度。为此还需对局部weight
decay项进行数值修正。如下公式，

.. math::

   \\nabla\_{k, t}^{'}=N \\nabla\_{k, t}+\\lambda w\_{t}, \\quad
   G\_{t}^{'}=\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}^{'}\\right)=N\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right)+N\\lambda w\_{t}, \\quad
   G\_{t}=\\frac{G\_{t}^{'}}{N}=\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right)+\\lambda w\_{t} \\tag{12}

具体做法为对局部梯度乘以卡数求得\ :math:`\nabla\_{k, t}^{'}`\ ，此时\ :math:`\lambda`\ 项则无需除以卡数，聚合梯度求得\ :math:`G\_{t}^{'}`\ 再对聚合梯度除以卡数得到\ :math:`G\_{t}`\ 即可。

上述策略已经在框架中实现，用户无须设置。

DGC 快速开始
~~~~~~~~~~~~

下文以单机八卡上训练ResNet50 为例子简单介绍 FleetX 中 DGC 的使用。 因为
8张 GPU 的通信都在同一节点内， 一般情况下梯度通信并不会成为训练的瓶颈，
这里只是以其为例子，介绍FleetX 中 DGC 参数的设置。

**注意**\ ：

-  硬件环境要求：
   DGC目前只支持GPU多卡及分布式collective训练，需要有相应的cuda、cuDNN、nccl环境。
-  Paddle环境要求： DGC只支持GPU，所以需GPU版本的Paddle。

添加依赖
^^^^^^^^

.. code:: python

   import os
   import fleetx as X
   import paddle.fluid as fluid
   import paddle.distributed.fleet.base.role_maker as role_maker
   import time
   import paddle.distributed.fleet as fleet

定义分布式模式并初始化
^^^^^^^^^^^^^^^^^^^^^^

通过\ ``X.parse_train_configs()``\ 接口，用户可以定义训练相关的参数，如：学习率、衰减率等。同时通过\ ``fleet.init()``\ 接口定义了分布式模型，下面代码中的\ ``is_collective=True``\ 表示采用集合通信的GPU分布式模式训练模型。

.. code:: python

   configs = X.parse_train_configs()
   role = role_maker.PaddleCloudRoleMaker(is_collective=True)
   fleet.init(role)

加载模型及数据
^^^^^^^^^^^^^^

用户可以通过\ ``X.applications``\ 接口加载我们预先定义好的模型，如：Resnet50、VGG16、BERT等。并使用定制化的data_loader加载模型，同时可以定义训练中使用的batch_size等参数。

.. code:: python

   model = X.applications.Resnet50()
   loader = model.load_imagenet_from_file("/pathto/ImageNet/train.txt")

DGC 相关策略
^^^^^^^^^^^^

这里假设：1252个step为一个epoch，前2个epochs使用正常dense通信，后3个epochs逐步提升稀疏度为99.9%

-  ``rampup_begin_step (int)``\ ：DGC(含预热训练)开始的 step
-  ``rampup_step (int)``\ ：DGC中预热训练持续的 step. 如果sparsity 是
   [0.75, 0.9375, 0.984375, 0.996, 0.999]，rampup_step 设成 100时， 在
   0~19 steps 时 sparsity=0.75，在 20~39 steps 时 sparsity=0.9375，
   以此类推。
-  ``sparsity (list[float])``\ ：稀疏度 threshold, (1 - current
   sparsity) % 的gradient 将会被 allreduce。

.. code:: python

   dist_strategy = fleet.DistributedStrategy()

   dist_strategy.lars = True
   dist_strategy.dgc_configs = {
       "rampup_begin_step": 1252*2,
       "rampup_step": 1252*3,
       "sparsity": [0.984375, 0.996, 0.999]
   }

   optimizer = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
   optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
   optimizer.minimize(model.loss)

开始训练
^^^^^^^^

这一部分和FleetX 中其他任务基本相同:

.. code:: python

   place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
   exe = fluid.Executor(place)
   exe.run(fluid.default_startup_program())

   total_time = 0
   for i, data in enumerate(data_loader()):
       start_time = time.time()
       cost_val = exe.run(paddle.static.default_main_program(),
                          feed=data,
                          fetch_list=[model.loss.name])
       end_time = time.time()
       total_time += (end_time - start_time)
       print(
           "worker_index: %d, step%d cost = %f, total time cost = %f, step per second: %f, speed: %f"
           % (fleet.worker_index(), i, cost_val[0], total_time,
              (i - 9) / total_time, 1 / (end_time - start_time))

运行训练脚本
^^^^^^^^^^^^

一行启动单机多卡分布式训练：

.. code:: sh

   fleetrun --gpus 0,1,2,3,4,5,6,7 resnet50_dgc.py

使用Local SGD 优化低带宽下分布式训练
------------------------------------

Local SGD 简介
~~~~~~~~~~~~~~

在使用 distributed SGD
进行数据并行的分布式训练时，常会遇到以下两个问题：

-  分布式训练的吞吐会受到集群中慢节点（straggling
   node）和随机通信延迟的影响。
-  数据并行分布式增大了训练实际的batch size，过大的batch size
   会影响最终的训练精度。

local SGD
通过延长节点间同步的间隔(局部异步训练)来减轻慢节点的影响和减少通信频率，以此提升训练的吞吐速度；另一方面，为了减小相对于本地训练（小batch
size）的精度损失，\ `DON’T USE LARGE MINI-BATCHES, USE LOCAL
SGD <https://arxiv.org/abs/1808.07217>`__ 和 `ADAPTIVE COMMUNICATION
STRATEGIES TO ACHIEVE THE BEST ERROR-RUNTIME TRADE-OFF IN LOCAL-UPDATE
SGD <https://arxiv.org/abs/1810.08313>`__
分别提出了：\ ``post-local SGD`` 和
``自适应步长 (Adaptive Communication)``
策略，来减少参数同步频率降低带来的精度损失。

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

在local SGD 训练中，集群中的每个 worker 各自会独立的进行 H 个连续的 SGD
更新， 然后集群中的所有 worker 会进行通信，同步（averaging）所有 workers
上的参数。一个双 workers，同步间隙为3 iterations 的local
SGD过程如上图所示。黄绿两条路径表示两个 workers 各自的 local SGD
更新过程，中间的蓝色路径表示同步后的模型所在的位置。

local
SGD中的一个关键问题是如何确定参数同步的间隔(频率)，以到达训练吞吐和训练精度间更好的平衡：

-  增大参数同步的间隔可以减少 workers 间通信延迟的影响提高训练吞吐.
-  但增大同步间隔可能会造成最终训练精度的损失。
   `[1] <https://arxiv.org/abs/1708.01012>`__

post-local SGD 将训练过程分成两个阶段：第一阶段 wokers 间同步的间隔为 1
iteration，即同步SGD，来保证最终训练精度；在第二阶段增大同步间隔到固定常数
H iterations，来提升训练吞吐。其公式如下：

Adaptive Communication local SGD
通过动态的调整参数同步的间隔来尝试达到训练吞吐和精度间的更好的平衡。在训练初始或者上一段参数同步完成后，根据如下公式计算一下次参数同步的间隔（iteration）。详细的公式推导和参数定义请参考\ `原论文 <https://arxiv.org/abs/1808.07217>`__\ 。

Fleet 中实现了 ``Naive local SGD`` 和
``Adaptive Communication local SGD`` 两种策略。 中下文将给出 Fleet中
local SGD 的实践效果，并通过一个简单例子介绍如何在Fleet 中使用 local
SGD。

Fleet 效果
~~~~~~~~~~

试验设置

+---------+---------+---------+---------+-------+---------+---------+
| model   | dataset | local   | cluster | dtype | warming | l       |
|         |         | batch   |         |       | up      | earning |
|         |         | size    |         |       |         | rate    |
|         |         |         |         |       |         | decay   |
+=========+=========+=========+=========+=======+=========+=========+
| r       | I       | 128     | 4 x 8 x | FP32  | 30      | pol     |
| esnet50 | magenet |         | V100    |       |         | ynomial |
+---------+---------+---------+---------+-------+---------+---------+

试验结果

========== ======= ====== ======
local step qps     acc1   acc5
========== ======= ====== ======
1          8270.91 0.7579 0.9266
2          8715.67 0.7533 0.9265
4          8762.66 0.7551 0.9260
8          9184.62 0.7511 0.9239
16         9431.46 0.7429 0.9206
ADACOMM    8945.74 0.7555 0.9270
========== ======= ====== ======

可以看到在 navie local SGD
（固定同步间隔）情况下，更新间隔越长训练的吞吐越高，但是模型的最终进度也会损失越大。
当使用 ADAPTIVE COMMUNICATION
策略后，训练在吞吐和精度间达到了一个更好的平衡。

local SGD 快速开始
~~~~~~~~~~~~~~~~~~

下文将以在单机8卡中训练 ResNet50 为例子简单介绍 Fleet 中 local SGD
的用法。 需要注意的是 单机八卡的通信都在同一节点内，
一般情况下参数同步并不会成为训练的瓶颈， 这里只是以其为例子，介绍Fleet
中 local SGD 参数的设置。

.. _添加依赖-1:

添加依赖
^^^^^^^^

.. code:: python

   import os
   import fleetx as X
   import paddle.fluid as fluid
   import paddle.distributed.fleet.base.role_maker as role_maker
   import time
   import paddle.distributed.fleet as fleet

.. _定义分布式模式并初始化-1:

定义分布式模式并初始化
^^^^^^^^^^^^^^^^^^^^^^

通过\ ``X.parse_train_configs()``\ 接口，用户可以定义训练相关的参数，如：学习率、衰减率等。同时通过\ ``fleet.init()``\ 接口定义了分布式模型，下面代码中的\ ``is_collective=True``\ 表示采用集合通信的GPU分布式模式训练模型。

.. code:: python

   configs = X.parse_train_configs()
   role = role_maker.PaddleCloudRoleMaker(is_collective=True)
   fleet.init(role)

.. _加载模型及数据-1:

加载模型及数据
^^^^^^^^^^^^^^

用户可以通过\ ``X.applications``\ 接口加载我们预先定义好的模型，如：Resnet50、VGG16、BERT等。并使用定制化的data_loader加载模型，同时可以定义训练中使用的batch_size等参数。

.. code:: python

   model = X.applications.Resnet50()
   loader = model.load_imagenet_from_file("/pathto/ImageNet/train.txt")

定义local SGD 相关策略
^^^^^^^^^^^^^^^^^^^^^^

用户首先需要定义paddle SGD 对象，并在SGD 对象中设置学习率参数。Fleet
local SGD 中只有两个用户设置参数 ``auto`` 和
``k_step``\ ，局部更新和参数同步都由框架自动完成：

-  在Naive local SGD 中： ``auto = Flase``\ ，
   用户需要设置一个固定的常数 ``k_step``
   作为训练过程中的全局参数更新间隔。
-  在 自适应步长 local SGD中： ``auto = True``\ ，
   用户需要设置\ ``k_step``
   作为第一次参数同步的间隔，之后的同步间隔将由上文中的公式动态确定，在学习率较大时，参数变化大，减小step，多进行通信从而保证快速收敛；在学习率较小时，参数变化小，增大step，减少通信次数，从而提升训练速度。
   需要注意的是自适应步长策略中，系统会默认限制最大的同步间隔为
   ``16 steps``\ ，当公式计算出的间隔大于16 时，按16 steps
   进行参数同步。

.. code:: python

   dist_strategy = fleet.DistributedStrategy()

   dist_strategy.localsgd = True
   dist_strategy.auto = True
   dist_strategy.localsgd_configs = {
                       "k_steps": 1,
                   }
   optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
   optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
   optimizer.minimize(model.loss)

.. _开始训练-1:

开始训练
^^^^^^^^

这一部分和FleetX 中其他任务基本相同:

.. code:: python

   place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
   exe = fluid.Executor(place)
   exe.run(fluid.default_startup_program())

   total_time = 0
   for i, data in enumerate(data_loader()):
       start_time = time.time()
       cost_val = exe.run(paddle.static.default_main_program(),
                          feed=data,
                          fetch_list=[model.loss.name])
       end_time = time.time()
       total_time += (end_time - start_time)
       print(
           "worker_index: %d, step%d cost = %f, total time cost = %f, step per second: %f, speed: %f"
           % (fleet.worker_index(), i, cost_val[0], total_time,
              (i - 9) / total_time, 1 / (end_time - start_time))

.. _运行训练脚本-1:

运行训练脚本
^^^^^^^^^^^^

一行启动单机多卡分布式训练：

.. code:: sh

   fleetrun --gpus 0,1,2,3,4,5,6,7 resnet50_localsgd.py
