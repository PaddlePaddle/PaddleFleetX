# word2vec benchmark on paddle

## 模型介绍
模型采用负采样 + skipgram，具体细节可参考https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/word2vec或者https://github.com/PaddlePaddle/Fleet/tree/develop/examples/word2vec

## 文件功能及介绍
* prepare_data.sh 数据下载及训练数据预处理
* preprocess.py 数据预处理文件，包含词表构建及数据过滤
* distribute_base.py 分布式、单机训练入口
* model.py 模型文件
* eval.py 测试文件
* py_reader_generator.py 数据读取
* dataset_generator.py  数据读取

## 使用方法
1. 数据处理：运行prepare.sh之后会在当前目录下得到三个文件夹，train_data用于训练，test_data用于测试，thirdparty目录下包含训练所需的词典文件test_build_dict，以及test_build_dict_word_to_id_
```
sh prepare_data.sh
```
2. 单机
* 训练，运行命令如下，首先创建相关目录，然后运行model.py文件。
```
mkdir -p model      ## 存放模型文件
mkdir -p result  ## 存放训练日志
mkdir -p log
python -u model.py --is_local=1 &> log/local.log &
```
* 测试
```
python eval.py --test_model_dir=model/
```
3. 分布式
* 训练
```
sh local_cluster.sh async
```
* 测试同单机

# paddle benchmark 实验结果
参数配置：
* learning rate: 1.0
* learning decay strategy: exponential_decay, decay_steps: 100000, decay_rate:0.999, staircase:True
* batch_size: 100
* embedding_size: 300
* nce: 5
* context window size: random[1, 5]
* threads: 1
* epochs: 5
* 数据预处理中down_sampling: 0.001

效果：

模式 | acc |  速度  
-|-|-
单机 | 0.595 | ~23小时 |
分布式异步 | / | / |
分布式同步 | / | / |
