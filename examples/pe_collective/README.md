# 简介
本示例基于MNIST数据集，展现如何使用Fleet API实现Paddle分布式训练和预测，同时展现如何在训练中使用Collective OP。

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
