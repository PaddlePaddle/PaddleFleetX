# Benchmark for CTR
An open sourced dataset in click through rate estimation task is used in this benchmark repo.
The task is to do a binary classification problem in which area under curve(auc) is used as evaluation metric. The code here mainly aims to provide reference scripts for users to test benchmark of **Multi-Thread on Single Machine** and **Distributed Training**. Model description and training arguments are from https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr.

# Local Training Benchmark
For click through estimation task, a practical approach is feature engineering. However, in this task, features are pre-defined in public dataset. We define a deep learning model to make use of existing sparse and dense features. In local training, multi-thread training is generally used. A benchmark of training throughput against differnet batch size and different training threads is provided below.

| batch v.s threads |  thread=11  |  thread=22  |
|:-----------------:|:-----------:|:-----------:|
|      batch=32     |  54595.47/s | 102253.81/s |
|      batch=64     |  79602.08/s | 146953.33/s |
|     batch=128     | 100372.17/s | 178673.38/s |
|     batch=256     | 110669.88/s | 201246.99/s |
|     batch=512     | 110595.13/s | 205129.46/s |
|     batch=1024    | 110564.76/s | 206454.75/s |
## Scripts for running the result on your server
```
sh get_data.sh
sh run_performance_benchmark.sh
```

# Evaluate Asynchronous Local Training Performance
This version of benchmark mainly uses the Hogwild! to parallelize training tasks between threads. The throught of the benchmark can be different given a batch size and thread number. Given high throughputs, we also care about the convergence properties of current model. Evaluations of auc on test set given models trained with different batch size and 40 threads are given below.

| thread=40 | batch=32 | batch=64 | batch=128 | batch=256 | batch=512 | batch=1024 |
|:---------:|:--------:|:--------:|:---------:|:---------:|:---------:|:----------:|
|  test auc |  0.7859  |  0.7942  |   0.7950  |   0.7943  |   0.7925  |    0.788   |

## Commands for running evaluation on single model
```
python infer_dataset.py --model_path 40_1024_models/epoch20.model/ --data_path test_data
```

# Distributed Training Benchmark
Since click through rate estimation is usually used on recommendation tasks and advertisement tasks. Big data is available on these tasks, we given distributed training benchmark based on internal used cluster so that users can reference on their own clusters.

|    batch=100    | 20worker20pserver11threads | 10worker10pserver11threads | 5worker5pserver11threads |
|:---------------:|:------------------------:|:--------------------------:|:--------------------------:|
|    sec/epoch    |            54            |             103            |             198            |
| ins/threads/sec |           3700           |            3860            |            4113            |
|     test auc    |         0.789627         |          0.793605          |          0.793794          |

|    batch=1000   | 20worker20pserver11threads | 10worker10pserver11threads | 5worker5pserver11threads |
|:---------------:|:------------------------:|:--------------------------:|:--------------------------:|
|    sec/epoch    |            42            |             81             |             159            |
| ins/threads/sec |           5023           |            5080            |            5220            |
|     test auc    |         0.774516         |          0.788851          |          0.794097          |

## script for running the task with 2worker2pserver on local machine
```
python launch.py --worker_num 5 --server_num 5 dist_ctr.py
```
You need to deploy the distributed training job on your cluster, the result is from a mpi cluster.

# Environment

## Local Machine
- Intel(R) Xeon(R) Gold 6271 CPU @ 2.60GHz
- cpu MHz : 2600.00
- cache size : 33792 KB
- cpu cores : 48
- paddle fluid version : release 1.5
- total memory : 256GB
- compile command : cmake -DCMAKE_INSTALL_PREFIX=./output/ -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON -DWITH_MKL=OFF -DWITH_GPU=OFF -DWITH_FLUID_ONLY=ON -DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python2.7/ -DPYTHON_LIBRARY=$PYTHONROOT/lib/libpython2.7.so -DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python2.7 ..

## Distributed Cluster
- AMD EPYC 7551P 32-Core Processor
- cpu MHz : 2000.00
- cache size : 512 KB
- cpu cores : 32
- paddle fluid version : release 1.5
- total memory : 256GB
- compile command : cmake -DCMAKE_INSTALL_PREFIX=./output/ -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON -DWITH_MKL=OFF -DWITH_GPU=OFF -DWITH_FLUID_ONLY=ON -DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python2.7/ -DPYTHON_LIBRARY=$PYTHONROOT/lib/libpython2.7.so -DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python2.7 ..

