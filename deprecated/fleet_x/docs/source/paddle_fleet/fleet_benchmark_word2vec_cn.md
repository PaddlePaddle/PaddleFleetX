## Word2vec模型性能基准

word2vec被广泛运用于NLP及推荐系统领域，同时也作为评价深度学习框架训练性能的重要模型之一，我们提供word2vec在1-billion数据集上的分布式训练的性能基准供用户参考,包含4、8、16、32台服务器联合训练。

### 软硬件配置情况

#### 基本版本信息
| 软硬件指标 | 具体配置 |
| ---- | ---- |
| 实例类型 | 纯CPU训练集群 |
| 操作系统 | Ubuntu 16.04 LTS with tests run via Docker |
| CPU | Intel(R) Xeon(R) CPU E5-2450 v2 @ 2.50GHz |
| 内存 | 128G |
| Paddle Github Commit | |
| FleetX Github Commit | |
| 硬盘类型 | 本地SSD硬盘 |
| 数据集 | 1-billion |
| 评估模型 | Work2vec |
| 复现代码地址 | [Word2vec-Benchmark](https://github.com/PaddlePaddle/FleetX/tree/develop/benchmark/paddle) |
| Python版本 | 3.7 |

### 性能测试方法

- 硬件资源
采用多机多进程训练，每一台服务器实例均启动一个pserver，一个trainer，每个trainer配置固定数量的线程，以实例数作为评价标准，评价 `4`, `8`, `16`, `32`情况下的性能基准。

- 训练超参数
批量大小（Batch Size）对训练性能影响最大，因此会对比不同批量大小下模型的训练吞吐。注意，改变批量大小通常需要调整优化算法，但为了对比公平，暂不对优化算法做调整>，即不考虑收敛的对比。

- 测试指标获取方法
当前主流的深度学习框架通常采用异步数据读取，由于训练开始前框架并没有提前开始读取数据，整个训练速度存在一定的IO瓶颈。我们的测试方法是统计一轮训练中总的字数，用一轮总字数除总耗时得到的平均吞吐作为单次任务的训练吞吐。为了避免硬件的波动（例如网络通信等）对性能的影响，我们会利用相同的配置进行7次运行，取中值。

### 基准测试结果

- 单位：`Images/s`，使用精度FP32，`DistributedStrategy`如下：

``` python
import paddle
import paddle.distributed.fleet as fleet

dist_strategy = fleet.DistributedStrategy()
dist_strategy.a_sync=True
dist_strategy.a_sync_configs = {"k_steps": 100}

```

| batch / node | 4 | 8 | 16 | 32 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| 100 | 55597.5 | 55082.37 | 53302.63 | 47280.91 |
