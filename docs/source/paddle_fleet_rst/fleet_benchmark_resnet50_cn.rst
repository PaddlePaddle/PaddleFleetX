Resnet50性能基准
----------------

Resnet50是当前视觉领域比较通用的预训练模型后端，同时也作为评价深度学习框架训练性能的最重要模型之一，我们提供Resnet50在ImageNet数据集上的性能基准供用户参考。

软硬件配置情况
~~~~~~~~~~~~~~

基本版本信息
^^^^^^^^^^^^

+----------------------+----------------------------------------------+
| 软硬件指标           | 具体配置                                     |
+======================+==============================================+
| 实例类型             | 百度X-Man 2.0                                |
+----------------------+----------------------------------------------+
| 单实例GPU            | 8x NVIDIA® Tesla® V100                       |
+----------------------+----------------------------------------------+
| 操作系统             | Ubuntu 16.04 LTS with tests run via Docker   |
+----------------------+----------------------------------------------+
| CPU                  | Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz    |
+----------------------+----------------------------------------------+
| 内存                 | 512G                                         |
+----------------------+----------------------------------------------+
| CUDA / CUDNN版本     | 10.1 / 7.6.5                                 |
+----------------------+----------------------------------------------+
| NCCL / DALI 版本     | 2.4.7 / 0.24.0                               |
+----------------------+----------------------------------------------+
| 多GPU实例互联信息    | InfiniBand 100 Gb/sec                        |
+----------------------+----------------------------------------------+
| Paddle Github Commit |                                              |
+----------------------+----------------------------------------------+
| FleetX Github Commit |                                              |
+----------------------+----------------------------------------------+
| 硬盘类型             | 本地SSD硬盘                                  |
+----------------------+----------------------------------------------+
| 数据集               | ImageNet                                     |
+----------------------+----------------------------------------------+
| 评估模型             | Resnet50                                     |
+----------------------+----------------------------------------------+
| 复现代码地址         | `Resn                                        |
|                      | et50-Benchmark <https://github.com/PaddlePad |
|                      | dle/FleetX/tree/develop/benchmark/paddle>`__ |
+----------------------+----------------------------------------------+
| Python版本           | 3.7                                          |
+----------------------+----------------------------------------------+

硬件拓扑
^^^^^^^^

.. code:: shell

   nvidia-smi topo -m

.. code:: shell

   GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    mlx5_0  CPU Affinity
   GPU0     X      NV2     NV2     NV1     NV1     NODE    NODE    NODE    NODE    0-23
   GPU1    NV2      X      NV1     NV1     NODE    NV2     NODE    NODE    NODE    0-23
   GPU2    NV2     NV1      X      NV2     NODE    NODE    NV1     NODE    NODE    0-23
   GPU3    NV1     NV1     NV2      X      NODE    NODE    NODE    NV2     NODE    0-23
   GPU4    NV1     NODE    NODE    NODE     X      NV2     NV2     NV1     NODE    0-23
   GPU5    NODE    NV2     NODE    NODE    NV2      X      NV1     NV1     NODE    0-23
   GPU6    NODE    NODE    NV1     NODE    NV2     NV1      X      NV2     NODE    0-23
   GPU7    NODE    NODE    NODE    NV2     NV1     NV1     NV2      X      NODE    0-23
   mlx5_0  NODE    NODE    NODE    NODE    NODE    NODE    NODE    NODE     X

   Legend:

   X    = Self
   SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
   NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
   PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
   PXB  = Connection traversing multiple PCIe switches (without traversing the PCIe Host Bridge)
   PIX  = Connection traversing a single PCIe switch
   NV#  = Connection traversing a bonded set of # NVLinks

性能测试方法
~~~~~~~~~~~~

-  硬件资源 采用多机多卡训练，以实例数 x 单实例GPU卡数作为评价标准，评价
   ``1 x 1``, ``1 x 8``, ``2 x 8``, ``4 x 8``\ 情况下的性能基准。

-  训练超参数 批量大小（Batch
   Size）对训练性能影响最大，因此会对比不同批量大小下模型的训练吞吐。注意，改变批量大小通常需要调整优化算法，但为了对比公平，暂不对优化算法做调整，即不考虑收敛的对比。

-  测试指标获取方法
   当前主流的深度学习框架通常采用异步数据读取，由于训练开始前框架并没有提前开始读取数据，整个训练速度存在一定的IO瓶颈。我们的测试方法是忽略前20个\ ``step``\ ，然后取后面100个step的平均吞吐作为单次任务的训练吞吐。为了避免硬件的波动（例如网络通信等）对性能的影响，我们会利用相同的配置进行7次运行，取中值。

基准测试结果
~~~~~~~~~~~~

-  单位：\ ``Images/s``\ ，使用精度FP32，\ ``DistributedStrategy``\ 如下：

.. code:: python

   import paddle
   import paddle.distributed.fleet as fleet
   dist_strategy = fleet.DistributedStrategy()
   exec_strategy = fluid.ExecutionStrategy()
   exec_strategy.num_threads = 2
   exec_strategy.num_iteration_per_drop_scope = 100
   dist_strategy.execution_strategy = exec_strategy
   build_strategy = fluid.BuildStrategy()
   build_strategy.enable_inplace = False
   build_strategy.fuse_elewise_add_act_ops = True
   build_strategy.fuse_bn_act_ops = True
   dist_strategy.build_strategy = build_strategy
   dist_strategy.nccl_comm_num = 1

============ ====== ======= ======= ========
batch / node 1 x 1  1 x 8   2 x 8   4 x 8    
============ ====== ======= ======= ========
32           335.43 2488.49 4629.71 9093.41  
64           353.38 2643.75 5325.44 10536.83 
128          368.11 2797.31 5635.98 11261.72 
============ ====== ======= ======= ========

-  单位：\ ``Images/s``\ ，使用自动混合精度Automatic Mixed
   Precision(AMP)进行训练，\ ``DistributedStrategy``\ 如下：

.. code:: python

   import paddle
   import paddle.distributed.fleet as fleet
   dist_strategy = fleet.DistributedStrategy()
   exec_strategy = fluid.ExecutionStrategy()
   exec_strategy.num_threads = 2
   exec_strategy.num_iteration_per_drop_scope = 100
   dist_strategy.execution_strategy = exec_strategy
   build_strategy = fluid.BuildStrategy()
   build_strategy.enable_inplace = False
   build_strategy.fuse_elewise_add_act_ops = True
   build_strategy.fuse_bn_act_ops = True
   dist_strategy.build_strategy = build_strategy
   dist_strategy.amp = True
   dist_strategy.nccl_comm_num = 1

============ ====== ======= ======== ========
batch / node 1 x 1  1 x 8   2 x 8    4 x 8
============ ====== ======= ======== ========
32           740.01 4467.82 8628.19  16970.01
64           919.95 6148.98 12071.29 23682.78
128          1018.3 7324.31 14342.03 28397.43
256          1096.5 8166.11 16189.79 32366.39
============ ====== ======= ======== ========

-  单位：\ ``Images/s``, 自动并行模式，\ ``DistributedStrategy``\ 如下：

.. code:: python

   import paddle.distributed.fleet as fleet
   dist_strategy = fleet.DistributedStrategy()
   dist_strategy.auto = True

为了获得更好的性能，我们默认打开了DALI进行数据IO，这里注意单机单开的自动并行开启的选项可能和多卡不同，因此加速比不具备参考意义。

============ ====== ======= ======== ========
batch / node 1 x 1  1 x 8   2 x 8    4 x 8    
============ ====== ======= ======== ========
32           666.38 4467.82 8711.69  19107.42 
64           761    6148.98 12076.77 24314.58 
128          890.03 6793.73 13514.66 27277.36 
256          938.57 7305.66 14599.55 29361.24 
============ ====== ======= ======== ========
