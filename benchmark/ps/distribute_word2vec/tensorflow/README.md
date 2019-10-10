# word2vec benchmark on tensorflow

## 模型介绍
模型采用负采样 + skipgram，具体细节可参考https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/word2vec或者https://github.com/PaddlePaddle/Fleet/tree/develop/examples/word2vec

## 文件功能及介绍
* prepare_data.sh 数据下载及训练数据预处理
* preprocess.py 数据预处理文件，包含词表构建及数据过滤
* net.py 训练及预测网络
* word2vec.py 单机word2vec训练代码
* word2vec_distribute.py 分布式word2vec训练代码
* local_cluster 分布式训练启动脚本
* reader.py 数据读取文件
* eval.py 测试文件

## 使用方法
1. 数据处理：运行prepare.sh之后会在当前目录下得到三个文件夹，train_data用于训练，test_data用于测试，thirdparty目录下包含训练所需的词典文件test_build_dict，以及test_build_dict_word_to_id_
```
sh prepare_data.sh
```
2. 单机非分布式
* 训练，运行命令如下，首先创建单机checkpoint和log保存目录，然后运行word2vec.py文件。此后可以通过命令```tail -f log/local/local.log```查看训练日志
```
mkdir -p output/local
mkdir -p log/local
python -u word2vec.py &> log/local/local.log &
```
* 预测，运行命令如下，首先创建预测日志和结果存放目录，然后执行eval.py进行预测，其中```task_mode```可以自行命名，用以区分不同任务模式，最终得到的预测日志文件和预测结果文件均以此命名，本示例中单机模式命名为```local```。```checkpoint_path```为训练checkpoint文件存放的路径，```result_path```则为保存结果文件的目录。最终预测任务产出为两个文件，一个是存放在evals/logs目录下以task_mode命名的日志文件，一个是存放在evals/results目录下以task_mode命名的结果json文件。
```
mkdir -p evals/logs
mkdir -p evals/results
python -u eval.py --task_mode=local --checkpoint_path=output/local --result_path=evals/results &> evals/logs/local.log &
```
3. 本地多进程模拟分布式
* 训练，运行命令如下，如果需要运行同步模式，则只需将async替换为sync，同时更改word2vec_distribute.py中的batch_size，tensorflow框架中同步(sync)的batch_size等于异步(async)batch_size/节点数
```
sh local_cluster.sh async
```
* 预测，用法同单机非分布式预测。
```
mkdir -p evals/logs
mkdir -p evals/results
python -u eval.py --task_mode=async --checkpoint_path=output/distribute/async --result_path=evals/results &> evals/logs/async.log &
```
## tensorflow benchmark 实验结果
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
单机 | 0.615 | ~8小时 |
分布式异步 | 0.595 | ~5.5小时 |
分布式同步 | / | / |
