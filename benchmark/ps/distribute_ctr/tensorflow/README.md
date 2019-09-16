# CTR(DNN) benchmark on tensorflow

## 模型介绍
参考https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr 或者 
https://github.com/PaddlePaddle/Fleet/tree/develop/examples/distribute_ctr

## 使用方法
1. 数据处理：数据处理文件为data_dnn_process.py，运行prepare.sh之后会在当前目录下得到两个文件夹，train_data用于训练，test_data用于测试
```
sh prepare_data.sh
```
2. 本地训练：包括分布式和非分布式训练两种
  * 本地非分布式训练，运行命令如下，首先创建checkpoint和log保存目录，然后运行ctr_dnn_local.py文件
```
mkdir -p output/checkpoint/local
mkdir -p log/local
python -u ctr_dnn_local.py &> log/local/local.log &
```
  * 本地多进程模拟分布式训练，运行命令如下，如果需要运行同步模式，则只需将async 替换为sync，同时更改ctr_dnn_distribute.py中的batch_size FLAG，tensorflow框架中同步(sync)的batch_size等于异步(async)batch_size/节点数
```
sh local_cluster.sh async
```
3. 预测
```
sh run_eval.sh
```

## tensorflow benchmark 实验结果
* 单机配置：11个线程，learning rate = 0.0001，batch_size = 1000
* 分布式配置：5x5(5个server,5个trainer)， 11个线程，learning rate = 0.0001， 同步batch_size = 200, 异步batch_size = 1000
* 实验效果
test auc 单机：0.7964 同步：0.7967 异步：0.7955
速度secs/epoch 单机：7480s(0.17 * 44000)，同步：4400s(0.10 * 44000)，异步：3916s(0.089 * 44000)
