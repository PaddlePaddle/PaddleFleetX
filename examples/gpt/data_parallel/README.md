# GPT 数据并行模型训练

数据并行是大规模深度学习训练中非常成熟和常用的并行模式。在数据并行模型训练中，训练任务被切分到多个进程(设备)上,每个进程维护相同的模型参数和相同的计算任务，但是处理不同的数据(batch data)；通过这种方式，同一全局数据(global batch)下的数据和计算被切分到了不同的进程，从而减轻了单个设备上的计算和存储压力。具体策略以及相关FleetAPI介绍可以参考以下教程：

- [数据并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/data_parallel/index_cn.html)


## 1.数据并行
当前GPT模型已适配数据并行，用户可以通过配置文件选择并行的维度。

```yaml
  Distributed:
    dp_degree: 8
```

其中参数含义：
- `dp_degree` 数据并行维度

## 2.运行方式


以单机8卡为例，通过``paddle.distributed.launch``启动多进程训练。

```shell
log_dir=dp8
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" run_pretrain.py \
    -c ./configs.yaml
```


执行日志：

```
[32m[2022-07-27 13:17:17,469] [    INFO][0m - global step 1, epoch: 0, batch: 0, loss: 11.266701698, avg_reader_cost: 0.24114 sec, avg_batch_cost: 3.90742 sec, speed: 0.26 step/s, ips_total: 16772 tokens/s, ips: 16772 tokens/s, learning rate: 5.55556e-09[0m
[32m[2022-07-27 13:17:19,467] [    INFO][0m - global step 2, epoch: 0, batch: 1, loss: 11.274262428, avg_reader_cost: 0.00020 sec, avg_batch_cost: 1.99697 sec, speed: 0.50 step/s, ips_total: 32818 tokens/s, ips: 32818 tokens/s, learning rate: 8.33333e-09[0m
[32m[2022-07-27 13:17:21,073] [    INFO][0m - global step 3, epoch: 0, batch: 2, loss: 11.266974449, avg_reader_cost: 0.00029 sec, avg_batch_cost: 1.60637 sec, speed: 0.62 step/s, ips_total: 40798 tokens/s, ips: 40798 tokens/s, learning rate: 1.11111e-08[0m
[32m[2022-07-27 13:17:22,692] [    INFO][0m - global step 4, epoch: 0, batch: 3, loss: 11.261226654, avg_reader_cost: 0.00017 sec, avg_batch_cost: 1.61802 sec, speed: 0.62 step/s, ips_total: 40504 tokens/s, ips: 40504 tokens/s, learning rate: 1.38889e-08[0m
[32m[2022-07-27 13:17:24,303] [    INFO][0m - global step 5, epoch: 0, batch: 4, loss: 11.268389702, avg_reader_cost: 0.00016 sec, avg_batch_cost: 1.61117 sec, speed: 0.62 step/s, ips_total: 40676 tokens/s, ips: 40676 tokens/s, learning rate: 1.66667e-08[0m
[32m[2022-07-27 13:17:25,915] [    INFO][0m - global step 6, epoch: 0, batch: 5, loss: 11.278966904, avg_reader_cost: 0.00016 sec, avg_batch_cost: 1.61185 sec, speed: 0.62 step/s, ips_total: 40659 tokens/s, ips: 40659 tokens/s, learning rate: 1.94444e-08[0m
[32m[2022-07-27 13:17:27,526] [    INFO][0m - global step 7, epoch: 0, batch: 6, loss: 11.280961037, avg_reader_cost: 0.00030 sec, avg_batch_cost: 1.61001 sec, speed: 0.62 step/s, ips_total: 40705 tokens/s, ips: 40705 tokens/s, learning rate: 2.22222e-08[0m
[32m[2022-07-27 13:17:29,127] [    INFO][0m - global step 8, epoch: 0, batch: 7, loss: 11.269421577, avg_reader_cost: 0.00016 sec, avg_batch_cost: 1.60079 sec, speed: 0.62 step/s, ips_total: 40940 tokens/s, ips: 40940 tokens/s, learning rate: 2.50000e-08[0m
[32m[2022-07-27 13:17:30,730] [    INFO][0m - global step 9, epoch: 0, batch: 8, loss: 11.264699936, avg_reader_cost: 0.00016 sec, avg_batch_cost: 1.60254 sec, speed: 0.62 step/s, ips_total: 40895 tokens/s, ips: 40895 tokens/s, learning rate: 2.77778e-08[0m
[32m[2022-07-27 13:17:32,333] [    INFO][0m - global step 10, epoch: 0, batch: 9, loss: 11.262663841, avg_reader_cost: 0.00015 sec, avg_batch_cost: 1.60261 sec, speed: 0.62 step/s, ips_total: 40893 tokens/s, ips: 40893 tokens/s, learning rate: 3.05556e-08[0m

```
