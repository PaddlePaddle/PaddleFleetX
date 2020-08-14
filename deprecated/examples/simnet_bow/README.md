#  Simnet-bow

## 模型简介

Simnet-bow是一种衡量文本相似度的神经网络模型, 通过在大规模点击数据上训练，达到衡量query-title语义匹配度的效果.
2013年，百度搜索成功上线基于海量用户反馈数据的SimNet-BOW语义匹配模型，实现了文本语义匹配特征的自动化提取，这也是深度学习技术首次成功应用于工业级搜索引擎中。此后，深度学习模型作为百度搜索智能化的核心驱动力得到不断完善。百度在数据特性、索引对象建模、模型结构等方面做了大量深入研究，先后自主研发并上线了SimNet-CNN、多域BOW、SimNet-RNN等多个模型，大幅提升了长冷搜索的排序效果。

模型设计可以参考：
>https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/similarity_net


## 使用方法
该代码库是为了向您提供基于Fleet的分布式Pserver模式使用示例。我们默认使用2X2，即2个PSERVER，2个TRAINER的组网方式进行训练。我们同时支持使用py_reader与Dataset方式进行高效的数据输入。主要组成为：

- 模型部分
   - model.py
   - argument.py
- 数据读取部分
   - py_reader_generator.py  
   - dataset_generator.py
- 分布式运行部分
   - distribute_base.py 
- 数据下载及程序入口
   - local_cluster.sh
   - get_data.sh

首先进行数据的下载。
```
sh get_data.sh
```
***
###### py_reader模式

开启参数服务器

```
sh local_cluster.sh pyreader sync ps
```
开启TRAINER端进行同步训练

```
sh local_cluster.sh pyreader sync tr
```
***
###### Dataset模式

开启参数服务器


```
sh local_cluster.sh dataset async ps
```
开启TRAINER端进行异步训练

```
sh local_cluster.sh dataset async tr
```

## 细节说明
- 训练结束后默认在0号TRAINER开启预测任务。
- 训练过程中的结果记录保存在./result。
- 训练得到的模型保存在./model。
- dataset模式目前仅支持运行在Linux环境下。
- 请确保您的Paddle fluid版本在1.5.0之上。
- 在每次结束运行后，建议使用下述命令，手动结束PSERVER的进程。请注意，该命令会同时结束其他python进程。
  >ps -ef|grep python|awk '{print $2}'|xargs kill -9
- 请根据自身系统选择bash命令替换sh





