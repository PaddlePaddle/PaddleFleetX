#  Simnet-bow

## 模型简介
本例实现了基于Fleet的分布式skip-gram模式的word2vector模型。为方便快速验证，采用了经典的text8样例数据集。全量数据可参考该实现：https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/word2vec


## 使用方法
该代码库是为了向您提供基于Fleet的分布式Pserver模式使用示例。我们默认使用2X2，即2个PSERVER，2个TRAINER的组网方式进行训练。我们同时支持使用py_reader与Dataset方式进行高效的数据输入。主要组成为：

- 模型部分
   - model.py
   - argument.py
   - preprocess.py
- 数据读取部分
   - py_reader_generator.py  
   - dataset_generator.py
- 分布式运行部分
   - distribute_base_.py 
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
- 训练中各个节点的日志保存在./log
- dataset模式目前仅支持运行在Linux环境下。
- 请确保您的Paddle fluid版本在1.5.0之上。
- 若频繁结束进程导致分布式训练不能启动时，请更改local_cluster.sh 中的端口，选择没有被占用的端口可以解决该问题。
- 不推荐在windows环境下运行该程序




