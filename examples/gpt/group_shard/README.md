# GPT 分组切分并行模型训练

当模型参数达到百亿或者千亿时， 传统的数据并行训练可能会遇到显存瓶颈。 在数据并行训练中，每个gpu worker 都有一份完整模型参数和优化器状态副本。 《[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)》指出在每个GPU 上都保存一份模型参数和优化器状态副本是冗余的。 我们可以通过将上述参数和副本划分到不同GPU 中， 在每个GPU 只保存部分副本，来减少每张GPU上显存的占用，从而可以支持更大模型的训练。具体策略以及相关FleetAPI介绍可以参考以下教程：

- [分组切分并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/group_sharded_parallel_cn.html)


## 1.分组切分并行
当前GPT模型已适配分组切分并行，用户可以通过配置文件选择并行维度和切分策略。

```yaml
  Distributed:
    sharding:
      sharding_degree: 8
      sharding_stage: 2
      sharding_offload: False
```

其中参数含义：
- `sharding_degree` 分组切分并行维度
- `sharding_stage` 切分策略。`2`表示切分梯度和优化器状态，`3`表示在上述策略基础上再切分前向参数
- `sharding_offload` CPU offload策略

## 2.运行方式


以单机8卡为例，通过``paddle.distributed.launch``启动多进程训练。

```shell
log_dir=sharding8
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" run_pretrain.py \
    -c ./configs.yaml
```


执行日志：

```
[32m[2022-07-27 08:53:11,036] [    INFO][0m - global step 1, epoch: 0, batch: 0, loss: 11.642805099, avg_reader_cost: 0.23430 sec, avg_batch_cost: 11.67036 sec, speed: 0.09 step/s, ips_total: 175 tokens/s, ips: 22 tokens/s, learning rate: 6.25000e-09[0m
[32m[2022-07-27 08:53:13,144] [    INFO][0m - global step 2, epoch: 0, batch: 1, loss: 11.620714188, avg_reader_cost: 0.00040 sec, avg_batch_cost: 2.10704 sec, speed: 0.47 step/s, ips_total: 972 tokens/s, ips: 121 tokens/s, learning rate: 9.37500e-09[0m
[32m[2022-07-27 08:53:13,540] [    INFO][0m - global step 3, epoch: 0, batch: 2, loss: 11.711474419, avg_reader_cost: 0.00016 sec, avg_batch_cost: 0.39656 sec, speed: 2.52 step/s, ips_total: 5164 tokens/s, ips: 646 tokens/s, learning rate: 1.25000e-08[0m
[32m[2022-07-27 08:53:13,836] [    INFO][0m - global step 4, epoch: 0, batch: 3, loss: 11.773808479, avg_reader_cost: 0.00019 sec, avg_batch_cost: 0.29522 sec, speed: 3.39 step/s, ips_total: 6937 tokens/s, ips: 867 tokens/s, learning rate: 1.56250e-08[0m
[32m[2022-07-27 08:53:14,150] [    INFO][0m - global step 5, epoch: 0, batch: 4, loss: 11.698161125, avg_reader_cost: 0.00029 sec, avg_batch_cost: 0.31358 sec, speed: 3.19 step/s, ips_total: 6531 tokens/s, ips: 816 tokens/s, learning rate: 1.87500e-08[0m
[32m[2022-07-27 08:53:14,433] [    INFO][0m - global step 6, epoch: 0, batch: 5, loss: 11.689817429, avg_reader_cost: 0.00034 sec, avg_batch_cost: 0.28225 sec, speed: 3.54 step/s, ips_total: 7256 tokens/s, ips: 907 tokens/s, learning rate: 2.18750e-08[0m
[32m[2022-07-27 08:53:14,734] [    INFO][0m - global step 7, epoch: 0, batch: 6, loss: 11.665119171, avg_reader_cost: 0.00023 sec, avg_batch_cost: 0.29825 sec, speed: 3.35 step/s, ips_total: 6867 tokens/s, ips: 858 tokens/s, learning rate: 2.50000e-08[0m
[32m[2022-07-27 08:53:15,015] [    INFO][0m - global step 8, epoch: 0, batch: 7, loss: 11.673336983, avg_reader_cost: 0.00024 sec, avg_batch_cost: 0.28085 sec, speed: 3.56 step/s, ips_total: 7292 tokens/s, ips: 912 tokens/s, learning rate: 2.81250e-08[0m
[32m[2022-07-27 08:53:15,295] [    INFO][0m - global step 9, epoch: 0, batch: 8, loss: 11.724355698, avg_reader_cost: 0.00022 sec, avg_batch_cost: 0.27952 sec, speed: 3.58 step/s, ips_total: 7327 tokens/s, ips: 916 tokens/s, learning rate: 3.12500e-08[0m
[32m[2022-07-27 08:53:15,577] [    INFO][0m - global step 10, epoch: 0, batch: 9, loss: 11.674280167, avg_reader_cost: 0.00021 sec, avg_batch_cost: 0.28149 sec, speed: 3.55 step/s, ips_total: 7276 tokens/s, ips: 909 tokens/s, learning rate: 3.43750e-08[0m

```
