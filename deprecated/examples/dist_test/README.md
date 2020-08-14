# 简介
本示例基于MNIST数据集，展现如何使用Fleet API实现Paddle分布式训练和预测。

我们在文件`train.py`中通过`args.distributed`配置项显式呈现了单机单卡代码如何使用Fleet API合入分布式能力。并且，**为了更加清晰地展现单卡训练和分布式训练在配置上的关键区别，避免分布式的配置分散不易阅读，我们将`args.distributed`配置项限制仅在主函数`main`里使用，以便于读者快速借鉴**。

即在`main`函数中，与分布式训练相关的代码要么在`if args.distributed`条件下执行，要么代入`args.distributed`参数。而其他可以在单卡训练和分布式训练中共享的代码均与此参数无关。

本示例需要基于paddle 1.8及以上版本运行，主要是分布式预测部分依赖1.8版本的特性。如果移除分布式预测代码，分布式训练部分的代码可以在更早版本的Paddle上运行。

# 文件结构

我们按照常见的方式创建了四个文件：

* `model.py`: 模型网络结构描述，含训练网络和测试网络
* `utils.py`: 工具函数
* `train.py`: 组建训练和评估流程的主函数
* `dist_run.sh`: 用于便捷启动分布式训练的脚本

建议从`train.py`开始阅读，快速了解Paddle分布式训练的配置过程。

# 执行方法

## 单机单卡

``` code::bash
python train.py
```

## 单机多卡

``` code::bash
sh dist_run.sh
```

## 多机多卡

假设有两台主机，ip地址分别为192.168.0.1和192.168.0.2，分别在两台主机上执行下述命令即可：

在ip地址为192.168.0.1的主机上：

``` code::bash
current_node_ip=192.168.0.1
cluster_node_ips=192.168.0.1,192.168.0.2

CUDA_VISIBLE_DEVICES=0,1 \
python -m paddle.distributed.launch \
    --selected_gpus=0,1 \
    --log_dir=mylog \
    --cluster_node_ips=$cluster_node_ips \
    --node_ip=$current_node_ip \
    train.py --distributed
```

或直接执行

```
sh multinodes/node1_run.sh
```

在ip地址为192.168.0.2的主机上：

``` code::bash
current_node_ip=192.168.0.2
cluster_node_ips=192.168.0.1,192.168.0.2

CUDA_VISIBLE_DEVICES=0,1 \
python -m paddle.distributed.launch \
    --selected_gpus=0,1 \
    --log_dir=mylog \
    --cluster_node_ips=$cluster_node_ips \
    --node_ip=$current_node_ip \
    train.py --distributed
```

或直接执行

```
sh multinodes/node2_run.sh
```

## 基于PaddleCloud执行多机多卡

假如有条件使用PaddleCloud集群，可以按照下述步骤提交任务：

1. 更新pcloud/run.sh配置中的AK/SK配置
2. 更新pcloud/job.cfg配置中的fs_name、fs_ugi和output_path
3. 执行下述命令：

```
cd pcloud && sh run.sh
```

注：2020年7月15日以后PaddleCloud才支持Paddle-1.8.2，在此时间之后笔者会再次验证此流程

# 分布式训练的数据划分

分布式训练一般情况下会将数据集划分为`nrank`份，每个`rank`训练属于自己的那一份数据。

共有三种常见的数据集组织方式：

1. 【推荐】一个文件包含一个样本，整个数据集由一个文件列表组成
2. 一个文件包含多个样本，整个数据集由一个文件列表组成
3. 一个文件包含多个样本，这个文件就是整个数据集，本示例就是这种数据组织

相应的切分方案是：

1. 将文件列表均匀拆分为`nranks`份
2. 将文件列表拆分为`nranks`份，并确保每个份数据中都包含相同数量的样本，否则可能会导致训练阻塞
3. 本地解析数据集，然后根据自身的`rank`号选择需要训练的数据。在这种条件下，不可避免地要进行冗余的数据解析

目前，避免样本不均匀分割的一种做法是将尾部数量少于`nranks`数的样本遗弃。另一种方法是每个`rank`都对整个数据集进行全局打散后训练。后者可以使各个`rank`的样本数量保持一致，但需要注意，此方法一遍`pass`训练的样本数量等同于前者`nranks`遍`pass`训练的数量。更进一步，如果我们预先知道我们需要训练的总`pass`数N，我们甚至可以在每个`rank`中复制N次数据集，然后全局打散后训练，这样可以更有效地利用数据集，这种策略唯一的代价是会丢失每个`pass`的边界信息。

我们可以将上述两种策略区分为无放回抽样和有放回抽样。

# 分布式预测基本原理

本示例需要进行分布式预测的指标是Accuracy。我们通过分布式计算Accurary为例讲解分布式预测的基本思路。

首先，使用上一小节的方案将测试数据切分到各个卡上，然后各个卡独立预测分配到本卡的数据，得到一个局部的True Positive值（正确预测样本数）和对应的权重（总样本数）。然后通过集合通信函数将各个卡的True Positive值和权重分别加和，最后计算```Accuracy=TruePositive/权重```求得全局的Accuracy。
