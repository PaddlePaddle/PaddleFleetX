
# 概述

**Fleet** is High-Level API for distributed training in PaddlePaddle. The name of **Fleet** means that a large crowd of ships working together to finish a large scale job. The design of Fleet makes a trade-off between easy-to-use and algorithmic extensibility and is highly efficient. First, a user can shift from local machine paddlepaddle code to distributed code  **within ten lines of code**. Second, different algorithms can be easily defined through **distributed strategy**  through Fleet API. Finally, distributed training is **extremely fast** with Fleet and just enjoy it.

**Note: all the examples here should be replicated from develop branch of Paddle**

## 目录
- [CPU分布式训练(Transplier)简介]()
- [Fleet API 介绍及使用]()
- [分布式CTR-DNN从零开始]()
- [分布式WORD2VEC从零开始]()
- [CPU分布式训练(Transplier)最佳实践]()
- [CPU分布式训练(Transplier)常见问题]()

### CPU分布式训练(Transplier)部分模型效果

<p align="center">
<img align="center" src="../../images/fleet_ps_benchmark_refine.png" height="270px" width="940px">
<p>


## 更多模型示例
- [Click Through Estimation](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/ctr)

- [Distribute CTR](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/distribute_ctr)

- [DeepFM](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/deepFM)

- [Semantic Matching](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/simnet_bow)

- [Word2Vec](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/word2vec)
