# Benchmark for Semantic Matching
An real world dataset in semantic matching is used in this benchmark repo.
The task is to do a semantic matching problem that a user needs to learn a matching function between two contextualized entities, for example, query and document. The code here mainly aims to provide reference scripts for users to test benchmark of **Multi-Thread on Single Machine** and **Distributed Training**.

# Local Training Benchmark
For semantic matching problem, a practical approach to learn the matching function is to use pairwise ranking loss(TODO: guru4elephant, add reference here). We build a very simple model for semantic matching that a multiple layers perceptron is used to model query and document side, and cosine similarity is computed with high level semantic representation vectors. For each positive pair, a negative pair is selected from the data. For the stability of benchmark, we do not use sampling strategy in choosing the negative pairs, although negative sampling stratigies are often used in practical scenarios. In local training, multi-thread training is commonly used. A benchmark of training throughput against different batch size and different working threads is provided below.

| batch v.s threads |  thread=11   |  thread=22  |
|:-----------------:|:------------:|:-----------:|
|      batch=32     |  340149.72/s | 179388.28/s |
|      batch=64     |  291731.59/s | 179293.29/s |
|     batch=128     |  359694.84/s | 235795.78/s |
|     batch=256     |  404779.02/s | 241717.98/s |
|     batch=512     |  410650.31/s | 219746.01/s |
|     batch=1024    |  398619.27/s | 237655.32/s |

## Scripts for running the result on your server
```
sh get_data.sh
python simnet_bow_benchmark.py
```

# Distributed Training Benchmark
Semantic Matching models are usually applied in web search, recommendation and advertisement. Big data is available on these tasks, we given distributed training benchmark based on internal used cluster so that users can reference on their own clusters.

|    batch=100    | 20worker20pserver11threads | 10worker10pserver11threads | 5worker5pserver11threads |
|:---------------:|:------------------------:|:--------------------------:|:--------------------------:|
|    sec/epoch    |            54            |             103            |             198            |
| ins/threads/sec |           3700           |            3860            |            4113            |

|    batch=1000   | 20worker20pserver11threads | 10worker10pserver11threads | 5worker5pserver11threads |
|:---------------:|:------------------------:|:--------------------------:|:--------------------------:|
|    sec/epoch    |            42            |             81             |             159            |
| ins/threads/sec |           5023           |            5080            |            5220            |

## script for running the task with 2worker2pserver on local machine
```
python launch.py --worker_num 2 --server_num 2 dist_simnet_bow.py
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

