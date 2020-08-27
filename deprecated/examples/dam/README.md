# 简介
本示例基于[这项工作](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2018-DAM)，使用Fleet API实现深度注意力机制模型（Deep Attention Matching Network）的分布式训练。

# 文件结构

我们按照常见的方式创建了几个文件：

* `train.py`: 组建训练和评估流程的主函数
* `model.py`: 模型网络结构描述，含训练网络和测试网络
* `layers.py`: 一些结构化的网络结构模块
* `evaluation.py`: 对预测值进行后处理的模块
* `dataloader.py`: 数据读取模块
* `utils.py`: 工具函数
* `config.py`: 命令行参数配置文件
* `data/download_data.sh`: 用于获取Ubuntu和Douban数据集
* `dist_run.sh`: 用于便捷启动分布式训练的脚本
* `test.sh`: 用于便捷启动单机预测的脚本
* `model_check.py`: 用于检查CUDA是否可用


# 执行方法

## Paddle 版本

```text
1.8.3
```

## 获取数据

``` code::bash
cd data
sh download_data.sh
```

## 单机单卡

``` code::bash
sh run.sh
```

## 单机多卡

``` code::bash
sh dist_run.sh
```
