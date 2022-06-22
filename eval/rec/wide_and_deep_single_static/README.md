# wide&deep

本教程使用Paddle静态图，在单机环境下训练wide&deep模型。

## 模型简介
[《Wide & Deep Learning for Recommender Systems》](https://arxiv.org/pdf/1606.07792.pdf)是Google 2016年发布的推荐框架，wide&deep设计了一种融合浅层（wide）模型和深层（deep）模型进行联合训练的框架，综合利用浅层模型的记忆能力和深层模型的泛化能力，实现单模型对推荐系统准确性和扩展性的兼顾。
具体的模型组网参见model.py脚本。

## 数据准备
数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。训练集包含一段时间内Criteo的点击流量。
每一行数据格式如下所示：
```
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。  
在data目录下为您准备了快速运行的示例数据，数据处理参见reader.py脚本。

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 


## 快速开始
在该目录下执行如下命令即可运行单机静态图版本的wide_and_deep模型训练 .
```bash
python trainer.py
```

