# CTR(DNN) benchmark on paddle

## 模型介绍
参考https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr 或者 
https://github.com/PaddlePaddle/Fleet/tree/develop/examples/distribute_ctr

## 文件及其功能介绍
* get_data.sh 数据下载
* model.py 模型文件
* distribute_base.py 训练、预测文件
* py_reader_generator.py 数据读取
* dataset_generator.py 数据读取
* eval.py 预测
* local_cluster.sh 分布式训练脚本

## 使用方法
1. 数据下载
```
sh get_data.sh
```
2. 单机非分布式
* 训练
```
mkdir -p model
mkdir -p result
mkdir -p log
python -u model.py --is_local=1 --is_dataset_train=True &> log/local.log &      # train from dataset 
python -u model.py --is_local=1 --is_pyreader_train=True &> log/local.log &     # train from pyreader
```
* 预测
```
python eval.py --test_model_dir=model/
```
3. 本地多进程模拟分布式
* 训练，运行命令如下，如果需要运行同步模式，则只需将async替换为sync，同时更改argument.py中的batch_size
paddle框架中同步(sync)的batch_size等于异步(async)batch_size/节点数/线程数
```
sh local_cluster.sh pyreader async
```
* 预测，同单机非分布式。

## tensorflow benchmark 实验结果

1. 单机配置：11个线程，learning rate = 0.0001，batch_size = 1000
2. 分布式配置：5x5(5个server,5个trainer)， 11个线程，learning rate = 0.0001， 同步batch_size = 20, 异步batch_size = 1000
3. 实验效果

模式 | auc 
-|-
单机pyreader | 0.796049 |
单机dataset | 0.796044 |
同步pyreader | 0.796499 | 
半异步pyreader | 0.794865 | 
异步pyreader | 0.794999 |
异步dataset | 0.794433 |
