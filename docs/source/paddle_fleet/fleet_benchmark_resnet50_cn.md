## Resnet50性能基准

Resnet50是当前视觉领域比较通用的预训练模型后端，同时也作为评价深度学习框架训练性能的最重要模型之一，我们提供Resnet50在ImageNet数据集上的性能基准供用户参考。

### 软硬件配置情况

| 软硬件指标 | 具体配置 |
| 实例类型 | 百度X-Man 2.0 |
| 单实例GPU | 8x NVIDIA® Tesla® V100 |
| 操作系统 | Ubuntu 16.04 LTS with tests run via Docker |
| CUDA / CUDNN版本 | 10.1 / 7.6.5 |
| NCCL / DALI 版本 | 2.4.7 / 0.24.0 |
| Paddle Github Commit | |
| FleetX Github Commit | |
| 硬盘类型 | 本地SSD硬盘 |
| 数据集 | ImageNet |
| 评估模型 | Resnet50 |
| 复现代码地址 | [Resnet50-Benchmark]() |

### 性能测试方法

- 硬件资源
采用多机多卡训练，以实例数 x 单实例GPU卡数作为评价标准，评价 `1 x 1`, `1 x 8`, `2 x 8`, `4 x 8`, `8 x 8`情况下的性能基准。

- 训练超参数
批量大小（Batch Size）对训练性能影响最大，因此会对比不同批量大小下模型的训练吞吐。注意，改变批量大小通常需要调整优化算法，但为了对比公平，暂不对优化算法做调整，即不考虑收敛的对比。

- 测试指标获取方法
当前主流的深度学习框架通常采用异步数据读取，由于训练开始前框架并没有提前开始读取数据，整个训练速度存在一定的IO瓶颈。我们的测试方法是忽略前20个`step`，然后取后面100个step的平均吞吐作为单次任务的训练吞吐。为了避免硬件的波动（例如网络通信等）对性能的影响，我们会利用相同的配置进行7次运行，取中值。

### 基准测试结果

- 单位：`Images/s`，使用精度FP32，`DistributedStrategy`如下：

``` python
import paddle.distributed.fleet as fleet
dist_strategy = fleet.DistributedStrategy()
dist_strategy.conv_workspace_size_limit = 4000
dist_strategy.cudnn_batchnorm_spatial_persistent = True
dist_strategy.fuse_grad_size_in_MB = 16
dist_strategy._fuse_grad_size_in_TFLOPS = 50.0
dist_strategy.cudnn_exhaustive_search = True
dist_strategy.sync_nccl_allreduce = True
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

```

| batch / node | 1 x 1 | 1 x 8 | 2 x 8 | 4 x 8 | 8 x 8 |
| 32 | | 2097.20 | 3982.28 | | |
| 64 | | 2386.88 | 4514.30 | | |
| 128 | | 2506.47 | 4913.35 | | |
| 256 |  | - | - | - | - |

- 单位：`Images/s`，使用精度AMP，`DistributedStrategy`如下：

``` python
import paddle.distributed.fleet as fleet
dist_strategy = fleet.DistributedStrategy()
dist_strategy.conv_workspace_size_limit = 4000
dist_strategy.cudnn_batchnorm_spatial_persistent = True
dist_strategy.fuse_grad_size_in_MB = 16
dist_strategy._fuse_grad_size_in_TFLOPS = 50.0
dist_strategy.cudnn_exhaustive_search = True
dist_strategy.sync_nccl_allreduce = True
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
dist_strategy.amp = True
```

| batch / node | 1 x 1 | 1 x 8 | 2 x 8 | 4 x 8 | 8 x 8 |
| 32 | | 4681.06 | 8524.23 | 16970.01 | |
| 64 | | 6259.26 | 11785.65 | 23682.78 | |
| 128 | | 7313.09 | 14143.96 | 28397.43 | |
| 256 |  | 8203.78 | 16153.68 | 32366.39 | |

- 单位：`Images/s`, 自动并行模式，`DistributedStrategy`如下：

``` python
import paddle.distributed.fleet as fleet
dist_strategy = fleet.DistributedStrategy()
dist_strategy.auto = True

```

| batch / node | 1 x 1 | 1 x 8 | 2 x 8 | 4 x 8 | 8 x 8 |
| 32 | | 4796.12 | 9466.25 | - | 30607.08 |
| 64 | | 5977.24 | 11925.01 | - | |
| 128 | | 6725.57 | 13455.14 | - | |
| 256 |  | 7261.00 | - | - | - |