#  deepFM

## 模型简介

论文来源：

>@inproceedings{guo2017deepfm,
  title={DeepFM: A Factorization-Machine based Neural Network for CTR Prediction},
  author={Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li and Xiuqiang He},
  booktitle={the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI)},
  pages={1725--1731},
  year={2017}
}

模型设计可以参考：
>https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/deepfm

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
   - data/get_data.sh
   - data/preprocess.py

首先进行数据的下载。
```
sh data/get_data.sh
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
- 训练结束后默认在0号TRAINER开启预测任务
- 训练过程中的结果记录保存在./result
- 训练得到的模型保存在./model
- dataset模式目前仅支持运行在Linux环境下
- 请确保您的Paddle fluid版本在1.5.0之上
- 在每次结束运行后，建议使用下述命令，手动结束PSERVER的进程。请注意，该命令会同时结束其他python进程。
  >ps -ef|grep python|awk '{print $2}'|xargs kill -9
- 请根据自身系统选择bash命令替换sh
- data/preprocess.py为您提供数据预处理的参考方法




