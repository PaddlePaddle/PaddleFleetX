# CTR-DNN

本教程使用Paddle静态图，在单机环境下训练CTR-DNN模型。

## 模型简介
CTR(Click Through Rate)，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。
CTR-DNN模型的组网比较直观，本质是一个二分类任务，代码参考model.py。模型主要组成是一个Embedding层，四个FC层，以及相应的分类任务的loss计算.

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
在该目录下执行如下命令即可运行单机静态图版本的CTR-DNN模型训练 .
```bash
python trainer.py
```

