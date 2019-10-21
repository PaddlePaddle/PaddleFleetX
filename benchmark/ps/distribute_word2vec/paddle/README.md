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

## 数据准备
运行prepare.sh之后会在当前目录下得到三个文件夹，train_data用于训练，test_data用于测试，thirdparty目录下包含训练所需的词典文件test_build_dict，以及test_build_dict_word_to_id_
```
sh prepare_data.sh
```

## 单机训练
```
mkdir -p model      ## 存放模型文件
mkdir -p result  ## 存放训练日志
python -u model.py --is_local=1
```

## 分布式模式及运行方式
1. dataset全异步模式
```
sh local_cluster dataset async
```

2. pyreader全异步模式
```
sh local_cluster.sh pyreader async
```

3. geo-sgd全异步模式
```
# 需先将loca_cluster.sh中FLAGS_communicator_thread_pool_size的注释去掉
sh local_cluster.sh dataset geo_async
```

4. pyreader同步模式（速度特别慢，不推荐）
```
# 需要将model.py中use_doulbe_buffer设为True
sh local_cluster.sh pyreader sync
```

## 测试
```
python eval.py --test_model_dir=model/
```
model/目录下应包含多个形如trainer_0_epoch_*的模型目录
