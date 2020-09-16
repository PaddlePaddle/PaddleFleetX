数据并行同步训练实践
====================

同步训练简介
------------

许多研究表明深度学习的预训练受益于更多的数据\ `[1] <https://arxiv.org/abs/1311.2901>`__
`[2] <https://arxiv.org/abs/1409.1556>`__
`[3] <https://arxiv.org/abs/1312.6229>`__\ ，但更大的数据量也意味着更长的训练耗时，数据并行同步训练是一种加速大规模数据训练的方法，有\ **PServer**\ 和\ **Collective**\ 两种模式。

同步训练通过数据划分，将计算工作量（前向、反向）分布到GPU
集群中的每一个worker上， 提高整体计算吞吐。但参数更新(update)
的过程在两种模式中有所不同：

-  在\ ``PServer模式``\ 中，会启动多个pservers
   和多个trainers，每个pserver会保存一部分模型参数，并负责接收从trainer发送的梯度并更新这些模型参数；每个trainer
   会保存一份完整的模型，并使用一部分数据进行训练，然后向pserver发送梯度，最后从pserver拉取更新后的参数。
   pserver进程和trainer可以在不同的计算节点上，也可以在同一公用节点。一个分布式任务所需要的pserver进程个数通常需要根据实际情况调整，以达到最佳的性能，然而通常来说pserver的进程不会比trainer更多。

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

-  在\ ``Collective模式``\ 中，集群中只存在多个地位平等的trainers。
   每个trainer进程都保存一份完整的模型参数。 前向和反向中每个 trainer
   使用自己划分 (shard）的数据进行计算，得到对应的梯度；之后trainers
   之间通过 allreduce 等 Collective
   通信方式\ `[4] <https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/>`__
   同步梯度到所有trainers，最后每个 trainer
   使用同步后的梯度独立完成参数更新。

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

相交于异步训练,
同步训练的的优势在于Loss可以比较稳定的下降，缺点是整体速度的快慢取决于最慢的trainer.
因此在训练较为复杂的模型时，即模型训练过程中神经网络训练耗时远大于节点间通信耗时的场景下，推荐使用同步训练模式。

Fleet中 PServer模式使用 gRPC 通信，Collective模式使用 NCCL2 通信。

Fleet Collective 同步训练优化
-----------------------------

Fleet 支持在 GPU (CUDA 版本 >= 7.5) 服务器集群上完成高性能分布式训练。
用户可以通过\ ``环境变量``\ 和\ ``fleet.DistributedStrategy``
设置许多与训练性能策略相关参数。目前Fleet
为这些参数提供了一个较通用默认值，用户可以不去调整。但如果用户希望针对性调优分布式训练的性能，可以根据自身硬件和任务设置对应参数。

在进行性能优化时， 检查每项优化点并验证对应提升，最终获得最优性能。
一个简单的验证当前的训练程序是否需要进一步优化性能的方法，
是查看GPU的计算利用率，通常用 :code:``nvidia-smi``\ 命令查看。
如果GPU利用率较低，则可能存在较大的优化空间。

下文对性能影响较大，设置频率比较高的几个参数，详细的参数列表放在文末的附录中。

注意：
使用NCCL2模式分布式训练时，需要确保每个节点训练等量的数据，防止在最后一轮训练中任务不退出。通常有两种方式：

-  随机采样一些数据，补全分配到较少数据的节点上。（推荐使用这种方法，以训练完整的数据集）。
-  在python代码中，每个节点每个pass只训练固定的batch数，如果这个节点数据较多，则不训练这些多出来的数据。

OP融合
~~~~~~

将模型网络中顺序执行的多个OPs进行融合能够减少OP
调度的开销，提升训练速度。目前Fleet 中支持如下3种的OP 融合：

-  ``fuse_all_optimizer_ops``\ ：表明是否融合(fuse) 是否融合
   optimizer_op，仅对部分 optimizer 可用（SGD、Adam和Momentum）。
-  ``fuse_elewise_add_act_ops``\ ：表明是否融合(fuse)
   elementwise_add_op和activation_op。
-  ``fuse_bn_act_ops``\ ：表明是否融合(fuse) batch_norm_op 和
   activation_op。

通常使用这些策略都会使整体执行过程更快。

.. code:: python

   dist_strategy = fleet.DistributedStrategy()
   dist_strategy.fuse_all_optimizer_ops = True
   dist_strategy.fuse_bn_act_ops = True
   dist_strategy.fuse_elewise_add_act_ops = True

AllReduce融合
~~~~~~~~~~~~~

AllReduce
融合默认情况下会将同一layer中参数的梯度的多个AllReduce操作合并成一个。
比如对于 fluid.layers.fc
中有Weight和Bias两个参数，打开该选项之前，需要两次AllReduce操作；打开该选项之后，只用一次AllReduce
操作。这样可以减少梯度同步时的通信耗时。

此外，为支持更大粒度的参数梯度融合，Fleet
提供了以下两个选项，用户可以在训练程序运行前在环境变量中设置：

-  ``FLAGS_fuse_parameter_memory_size``:
   指定每个AllReduce操作的梯度字节数，如该参数等于16
   则每次AllReduce调用传输16MB的梯度。
   该参数的经验值为总通信量的十分之一。
-  ``FLAGS_fuse_parameter_groups_size``:
   指定每次AllReduce操作的最大层数，即到达该层数就进行AllReduce。如该参数等于50,
   则最多每50层做一次 fused AllReduce。

注意： AllReduce融合目前不支持sparse参数梯度。

.. code:: shell

   export FLAGS_fuse_parameter_memory_size=16 
   export FLAGS_fuse_parameter_groups_size=50

.. code:: python

   dist_strategy = fleet.DistributedStrategy()
   dist_strategy.fuse_all_reduce_ops=True

分层 AllReduce
~~~~~~~~~~~~~~

对于多机模式，针对小数据量的通信，Ring
AllReduce通信效率低，采用Hierarchical AllReduce可以缓解这一问题。

.. code:: python

   dist_strategy = fleet.DistributedStrategy()
   dist_strategy.use_hierarchical_allreduce = True
   dist_strategy.hierarchical_allreduce_inter_nranks = 8

选择通信模式和执行模式
~~~~~~~~~~~~~~~~~~~~~~

Fleet 使用多进程+NCCL2模式（collective）以获得更好的性能。
在多进程模式下，每台服务器的每个GPU卡都会对应启动一个训练进程，
集群中的所有进程之间会互相通信完成训练。以此方式最大限度的降低进程内部资源抢占的开销。
对比在单进程开启ParallelGraph方法，多进程模式不但可以获得更高性能，
而且无需考虑reader在多卡下io性能不足的问题，直接使用多进程提升数据读取IO效率。

使用ParallelGraph模式相对而言会减少多进程管理，并提升性能，而且可以无需修改代码，只需要开启下列开关即可：

.. code:: shell

   export FLAGS_enable_parallel_graph=1

如果是单机多卡模式，同样可以通过开启ParallelGraph来提升性能：

.. code:: shell

   export FLAGS_enable_parallel_graph=1
   export FLAGS_sync_nccl_allreduce=1

设置合适的nccl通信器数量
~~~~~~~~~~~~~~~~~~~~~~~~

nccl通信器数量 nccl_comm_num
可以加快GPU之间的通信效率，建议单机设置为1，多机设置为2。

.. code:: python

   dist_strategy = fleet.DistributedStrategy()
   dist_strategy.nccl_comm_num = 2

设置合适的CPU线程数
~~~~~~~~~~~~~~~~~~~

PaddlePaddle Fluid使用“线程池”
`[5] <https://en.wikipedia.org/wiki/Thread_pool>`__
模型调度并执行Op，Op在启动GPU计算之前，
通常需要CPU的协助，然而如果Op本身占用时间很小，“线程池”模型下又会带来额外的调度开销。
使用多进程模式时，如果神经网络的计算图
`[6] <https://en.wikipedia.org/wiki/Data-flow_diagram>`__
节点间有较高的并发度，
即使每个进程只在一个GPU上运行，使用多个线程可以更大限度的提升GPU利用率。

根据以往的经验，对于CPU任务，num_threads=2 \* ev_count
时性能较好，对于GPU任务，num_threads=4 \* dev_count
时性能较好。注意：线程池不是越大越好。

.. code:: python

   dist_strategy = fleet.DistributedStrategy()
   dist_strategy.thread_num = 3

提高网络的吞吐
~~~~~~~~~~~~~~

多节点训练时网络的带宽常常成为训练的瓶颈。我们在实测中发现，当\ **使用自动混合精度训练后，TCP
socket 的通信方式将成为训练速度的瓶颈， 使多节点训练无法充分利用 FLeet
混合精度计算带来的速度提升**\ 。 在我们实测中使用: 100Gb
网卡，\ ``RDMA``\ `[7] <https://docs.nvidia.com/cuda/gpudirect-rdma/index.html>`__
和
``InfiniBand``\ `[8] <https://zh.wikipedia.org/wiki/InfiniBand>`__\ 来提升网络带宽，使网络传输不会成为计算速度的瓶颈。
在开始训练前，需要正确设置以下 NCCL 环境变量使对应硬件设置生效：

==================== =============================================
Env Name             Description
==================== =============================================
NCCL_SOCKET_IFNAME   The RDMA device, e.g. eth2
NCCL_P2P_DISABLE     Set to 1 to disable P2P transfer between GPUs
NCCL_IB_DISABLE      Set to 1 to disable using RDMA
NCCL_IB_CUDA_SUPPORT Set to 1 to enable GPU Direct if supported
NCCL_DEBUG           Set debug level: VERSION, WARN, INFO
==================== =============================================

预先分配足够的显存
~~~~~~~~~~~~~~~~~~

通过设置环境变量 FLAGS_fraction_of_gpu_memory_to_use=0.7
设置预先分配的显存占比。
由于CUDA原生的显存分配cuMalloc和释放cuFree操作均是同步操作，非常耗时，因此
通过 设置 FLAGS_fraction_of_gpu_memory_to_use
成一个较大的值，比如0.7，可以显著地加速训练的速度。

0.7 是指 70%的显存会预先分配。设置的范围是0.0~1.0。注意，
设置成0.0会让每次显存分配都调用 cudaMalloc 这样会极大的降低训练性能。

.. code:: shell

   export FLAGS_fraction_of_gpu_memory_to_use=0.7

降低scope drop频率和fetch频率
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

减少scope drop和fetch频率，可以减少频繁的变量内存申请、释放和拷贝，
从而提升性能。

.. code:: python

   # 每 30 batch 之后清理一次临时变量
   dist_strategy = fleet.DistributedStrategy()
   dist_strategy.BuildStrategy = {'num_iteration_per_drop_scope': 30}

   # 降低fetch频率，每 30 batch fetch 一次训练输出
   for pass_id in xrange(PASS_NUM):
       batch_id = 0
       while True:
           if batch_id % 30 == 0:
               fetched = exe.run(fetch_list)
           else:
               exe.run([])

增大batch_size或使用设置通信频率（batch merge）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

分布式同步训练，跨节点通信或多或少会带来性能影响，增大训练的batch_size，
可以保持通信开销不变的情况下，增大计算吞吐从而降低通信在整个训练过程中的占比来提升总体的训练吞吐。

然而增大batch_size会带来同等比例的显存消耗提升，为了进一步的增大batch_size，Fluid提供“batch
merge”功能，
通过在一个GPU上串行计算多个小的batch并积累梯度，然后再执行多机多卡之间的通信，
此模式同样也可以被称为“可变通信频率“。使用batch
merge功能，在同样的模型， 可以极大的增加batch
size，提升多机训练的总吞吐。

使用 DALI reader
~~~~~~~~~~~~~~~~

数据读取的优化在GPU训练中至关重要，尤其在不断增加batch_size提升吞吐时，数据reader
可能成为训练速度的瓶颈。 Fleet 中可以使用 Nvidia
DALI\ `6 <https://docs.nvidia.com/deeplearning/dali/master-user-guide/docs/>`__
作为数据reader. 使用DALI的有点有：

-  使用GPU完成部分数据预处理，加速数据读取过程，减少 CPU 负担。
-  DALI 提供预取队列（perfetch
   queue）功能，让数据预处理和模型计算可以异步进行，减少模型计算对数据读取的等待。

.. code:: python

   import fleetx as X
   model = X.applications.Resnet50()
   loader = model.load_imagenet_from_file("/pathto/imagenet/train.txt", use_dali=True)

使用混合精度训练
~~~~~~~~~~~~~~~~

V100 GPU提供了 Tensor Core 可以在混合精度计算
场景极大的提升性能。使用混合精度计算的例子可以参考文档
` <https://todo/>`__

目前Paddle只提供在两个模型（ResNet, BERT）的混合精度计算实现并支持static
loss scaling，其他模型使用混合精度也 可以参考以上的实现完成验证。

Fleet 训练策略
--------------

DistributedStrategy
^^^^^^^^^^^^^^^^^^^

+-----------------+-----------------+-----------------+-----------------+
| Dist            | 类型            | 默认值          | 定义            |
| ributedStrategy |                 |                 |                 |
+=================+=================+=================+=================+
| auto            | bool            | False           | 自动            |
|                 |                 |                 | 化框架参数优化  |
+-----------------+-----------------+-----------------+-----------------+
| a_sync          | bool            | True            | 指示            |
|                 |                 |                 | 是否使用异步SGD |
|                 |                 |                 | 进行参          |
|                 |                 |                 | 数更新，仅在PS  |
|                 |                 |                 | erver模式中生效 |
+-----------------+-----------------+-----------------+-----------------+
| sync            | bool            | True            | 指示是          |
| _nccl_allreduce |                 |                 | 否在每个通信线  |
|                 |                 |                 | 程中中使用同步  |
|                 |                 |                 | allre           |
|                 |                 |                 | duce，仅在Colle |
|                 |                 |                 | ctive模式中生效 |
|                 |                 |                 | ，通常在使用同  |
|                 |                 |                 | 步allreduce后系 |
|                 |                 |                 | 统的开销会降低  |
+-----------------+-----------------+-----------------+-----------------+
| nccl_comm_num   | int             | 1               | nccl通信器数量. |
|                 |                 |                 | nccl通信器数量  |
|                 |                 |                 | nccl_comm_num   |
|                 |                 |                 | 可以加快GPU之   |
|                 |                 |                 | 间的通信效率，  |
|                 |                 |                 | 建议单机设置为  |
|                 |                 |                 | 1，多机设置为2  |
|                 |                 |                 | 。针对CPU线程数 |
|                 |                 |                 | num_threads     |
|                 |                 |                 | ，建议单机设置  |
|                 |                 |                 | 为1，多机设置为 |
|                 |                 |                 | nccl_comm_num   |
|                 |                 |                 | +1              |
+-----------------+-----------------+-----------------+-----------------+
| use_hierarc     | bool            | False           | 分级式allred    |
| hical_allreduce |                 |                 | uce，对于多机模 |
|                 |                 |                 | 式，针对小数据  |
|                 |                 |                 | 量的通信，Ring  |
|                 |                 |                 | AllReduc        |
|                 |                 |                 | e通信效率低，采 |
|                 |                 |                 | 用Hierarchical  |
|                 |                 |                 | AllReduce可     |
|                 |                 |                 | 以解决该问题。  |
+-----------------+-----------------+-----------------+-----------------+
| hiera           | int             | 1               | 在              |
| rchical_allredu |                 |                 | 分级式allreduc  |
| ce_inter_nranks |                 |                 | e，低层级groups |
|                 |                 |                 | 中的            |
|                 |                 |                 | r               |
|                 |                 |                 | ank数。一般等于 |
|                 |                 |                 | 单个GPU节点中的 |
|                 |                 |                 | GPU数           |
+-----------------+-----------------+-----------------+-----------------+
| sync_batch_norm | bool            | False           | 表示是否使      |
|                 |                 |                 | 用同步的批正则  |
|                 |                 |                 | 化，即在训练阶  |
|                 |                 |                 | 段通过多个设备  |
|                 |                 |                 | 同步均值和方差  |
|                 |                 |                 | 。当前的实现不  |
|                 |                 |                 | 支持FP16训练和C |
|                 |                 |                 | PU。并且目前\ * |
|                 |                 |                 | *仅支持**\ 仅在 |
|                 |                 |                 | 一台机器上进行  |
|                 |                 |                 | 同步式批正则。  |
+-----------------+-----------------+-----------------+-----------------+
| fuse            | bool            | True            | 默认情况下会    |
| _all_reduce_ops |                 |                 | 将同一layer中参 |
|                 |                 |                 | 数的梯度的AllR  |
|                 |                 |                 | educe操作合并成 |
|                 |                 |                 | 一个，比如对于  |
|                 |                 |                 | fluid.layers.fc |
|                 |                 |                 | 中有            |
|                 |                 |                 | Weight和Bias两  |
|                 |                 |                 | 个参数，打开该  |
|                 |                 |                 | 选项之后，原本  |
|                 |                 |                 | 需要两次AllRed  |
|                 |                 |                 | uce操作，现在只 |
|                 |                 |                 | 用一次AllReduce |
|                 |                 |                 | 操作。          |
+-----------------+-----------------+-----------------+-----------------+
| fuse_           | int             | 32              | 每个AllReduce操 |
| grad_size_in_MB |                 |                 | 作的梯度字节数  |
+-----------------+-----------------+-----------------+-----------------+
| fuse_grad       | int             | 20              | 指              |
| _size_in_TFLOPS |                 |                 | 定每次AllReduc  |
|                 |                 |                 | e操作的最大层数 |
|                 |                 |                 | ，即到达该层数  |
|                 |                 |                 | 就进行AllReduce |
+-----------------+-----------------+-----------------+-----------------+
| cudnn_ex        | bool            | True            | 表示是          |
| haustive_search |                 |                 | 否使用穷举搜索  |
|                 |                 |                 | 方法来选择卷积  |
|                 |                 |                 | 算法。在cuDNN中 |
|                 |                 |                 | 有两种搜索方法  |
|                 |                 |                 | ，启发式搜索和  |
|                 |                 |                 | 穷举搜索。穷举  |
|                 |                 |                 | 搜索尝试所有cu  |
|                 |                 |                 | DNN算法以选择其 |
|                 |                 |                 | 中最快的算法。  |
|                 |                 |                 | 此方法非常耗时  |
|                 |                 |                 | ，所选择的算法  |
|                 |                 |                 | 将针对给定的层  |
|                 |                 |                 | 规格进行缓存。  |
|                 |                 |                 | 一旦更改了      |
|                 |                 |                 | 图层规格（如bat |
|                 |                 |                 | ch大小，feature |
|                 |                 |                 | map大小），     |
|                 |                 |                 | 它将再次搜索。  |
+-----------------+-----------------+-----------------+-----------------+
| conv_works      | int             | 4000            | 用              |
| pace_size_limit |                 |                 | 于选择cuDNN卷积 |
|                 |                 |                 | 算法的工作区限  |
|                 |                 |                 | 制大小（单位为  |
|                 |                 |                 | MB）。cuDNN的内 |
|                 |                 |                 | 部函数在这个内  |
|                 |                 |                 | 存限制范围内获  |
|                 |                 |                 | 得速度最快的匹  |
|                 |                 |                 | 配算法。通常，  |
|                 |                 |                 | 在较大的工作区  |
|                 |                 |                 | 内可以选择更快  |
|                 |                 |                 | 的算法，但同时  |
|                 |                 |                 | 也会显著增加内  |
|                 |                 |                 | 存空间。用户需  |
|                 |                 |                 | 要在内存和速度  |
|                 |                 |                 | 之间进行权衡。  |
+-----------------+-----------------+-----------------+-----------------+
| cudn            | bool            | True            | 表示是否在      |
| n_batchnorm_spa |                 |                 | batchnorm中使用 |
| tial_persistent |                 |                 | 新的批量标准化  |
|                 |                 |                 | 模式CUDNN_BATC  |
|                 |                 |                 | HNORM_SPATIAL_P |
|                 |                 |                 | ERSISTENT函数。 |
+-----------------+-----------------+-----------------+-----------------+

BuildStrategy
^^^^^^^^^^^^^

+-----------------+-----------------+-----------------+-----------------+
| BuildStrategy   | 类型            | 默认值          | 定义            |
+=================+=================+=================+=================+
| enable_seque    | bool            | False           | 如果            |
| ntial_execution |                 |                 | 设置为True，则  |
|                 |                 |                 | 算子的执行顺序  |
|                 |                 |                 | 将与算子定义的  |
|                 |                 |                 | 执行顺序相同。  |
+-----------------+-----------------+-----------------+-----------------+
| fuse_elew       | bool            | False           | 表明            |
| ise_add_act_ops |                 |                 | 是否融合(fuse)  |
|                 |                 |                 | elementwise_add |
|                 |                 |                 | _op和activation |
|                 |                 |                 | _op。这会使整体 |
|                 |                 |                 | 执行过程更快。  |
+-----------------+-----------------+-----------------+-----------------+
| fuse_bn_act_ops | bool            | False           | 表明            |
|                 |                 |                 | 是否融合(fuse)  |
|                 |                 |                 | batch_norm_op   |
|                 |                 |                 | 和              |
|                 |                 |                 | activation      |
|                 |                 |                 | _op。这会使整体 |
|                 |                 |                 | 执行过程更快。  |
+-----------------+-----------------+-----------------+-----------------+
| fuse_relu       | bool            | False           | 表明            |
| _depthwise_conv |                 |                 | 是否融合(fuse)  |
|                 |                 |                 | relu和          |
|                 |                 |                 | depthwise_conv  |
|                 |                 |                 | 2d，节省GPU内存 |
|                 |                 |                 | 并可能加速执行  |
|                 |                 |                 | 过程。此选项仅  |
|                 |                 |                 | 适用于GPU设备。 |
+-----------------+-----------------+-----------------+-----------------+
| fus             | bool            | False           | 表明            |
| e_broadcast_ops |                 |                 | 是否融合(fuse)  |
|                 |                 |                 | broadcast       |
|                 |                 |                 | ops。           |
|                 |                 |                 | 该选项指在Reduc |
|                 |                 |                 | e模式下有效，使 |
|                 |                 |                 | 程序运行更快。  |
+-----------------+-----------------+-----------------+-----------------+
| fuse_al         | bool            | False           | 表明            |
| l_optimizer_ops |                 |                 | 是否融合(fuse)  |
|                 |                 |                 | 是否融合        |
|                 |                 |                 | optimiz         |
|                 |                 |                 | er_op，仅对部分 |
|                 |                 |                 | optimizer       |
|                 |                 |                 | 可用            |
|                 |                 |                 | （SGD、Adam和M  |
|                 |                 |                 | omentum），可使 |
|                 |                 |                 | 程序运行更快。  |
+-----------------+-----------------+-----------------+-----------------+
| enable_inplace  | bool            | False           | 表明是          |
|                 |                 |                 | 否Op的输出复用O |
|                 |                 |                 | p输入的显存空间 |
|                 |                 |                 | ，优化显存占用  |
+-----------------+-----------------+-----------------+-----------------+
| ena             | bool            | True            | 在反向操作      |
| ble_backward_op |                 |                 | 和参数更新操作  |
| timizer_op_deps |                 |                 | 之间添加依赖，  |
|                 |                 |                 | 保证在所有的反  |
|                 |                 |                 | 向操作都运行结  |
|                 |                 |                 | 束之后才开始运  |
|                 |                 |                 | 行参数更新操作. |
|                 |                 |                 | 在              |
|                 |                 |                 | 多卡训练时，打  |
|                 |                 |                 | 开该选项可能会  |
|                 |                 |                 | 提升训练速度。  |
+-----------------+-----------------+-----------------+-----------------+
| cache_          | bool            | False           | unkown          |
| runtime_context |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+

ExecutionStrategy
^^^^^^^^^^^^^^^^^

+-----------------+-----------------+-----------------+-----------------+
| Ex              | 类型            | 默认值          | 定义            |
| ecutionStrategy |                 |                 |                 |
+=================+=================+=================+=================+
| num_threads     | int             | 1               | 表示当前        |
|                 |                 |                 | Executor        |
|                 |                 |                 | 的线程池(thread |
|                 |                 |                 | pool)的大小,    |
|                 |                 |                 | 此线            |
|                 |                 |                 | 程池可用来并发  |
|                 |                 |                 | 执行program中的 |
|                 |                 |                 | operator（算子  |
|                 |                 |                 | ，运算）。如果  |
|                 |                 |                 | num_threads=1   |
|                 |                 |                 | ，则所有        |
|                 |                 |                 | 的operator将一  |
|                 |                 |                 | 个接一个地执行  |
|                 |                 |                 | ，但在不同的pro |
|                 |                 |                 | gram重复周期(it |
|                 |                 |                 | erations)中执行 |
|                 |                 |                 | 顺序可能不同。  |
+-----------------+-----------------+-----------------+-----------------+
| num_iteration   | int             | 10              | 该选项表        |
| _per_drop_scope |                 |                 | 示间隔多少次迭  |
|                 |                 |                 | 代之后清理一次  |
|                 |                 |                 | 临时变量。模型  |
|                 |                 |                 | 运行过程中，生  |
|                 |                 |                 | 成的中间临时变  |
|                 |                 |                 | 量将被放到local |
|                 |                 |                 | execution       |
|                 |                 |                 | scope中，为了   |
|                 |                 |                 | 避免对临时变量  |
|                 |                 |                 | 频繁的申请与释  |
|                 |                 |                 | 放，通常将其设  |
|                 |                 |                 | 为较大的值（比  |
|                 |                 |                 | 如10或者100）。 |
+-----------------+-----------------+-----------------+-----------------+
| num_it          | int             | 3               | 它配置了当用户  |
| eration_per_run |                 |                 | 在python脚本中  |
|                 |                 |                 | 调用pe.run()时  |
|                 |                 |                 | 执行器会执行的  |
|                 |                 |                 | 迭代次数。Execu |
|                 |                 |                 | tor每次调用，会 |
|                 |                 |                 | 进行num_iterat  |
|                 |                 |                 | ion_per_run次训 |
|                 |                 |                 | 练，它会使整体  |
|                 |                 |                 | 执行过程更快。  |
+-----------------+-----------------+-----------------+-----------------+
| use             | bool            | False           | 当使用 PServer  |
| _thread_barrier |                 |                 | 模式时为 True   |
+-----------------+-----------------+-----------------+-----------------+

参考
----

https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/performance_improving/multinode_training_improving/dist_training_gpu.html
https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/howto/training/cluster_howto.html
https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/best_practice/dist_training_gpu.html#id17
https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/best_practice/training_best_practice.html
https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/BuildStrategy_cn.html
https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/distributed_training/cluster_quick_start.html
https://paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/howto/training/cluster_quick_start.html
https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_guides/low_level/distributed/sync_training.html#nccl2
