
# Fleet

**Fleet** is High-Level API for distributed training in PaddlePaddle. The name of **Fleet** means that a large crowd of ships working together to finish a large scale job. The design of Fleet makes a trade-off between easy-to-use and algorithmic extensibility and is highly efficient. First, a user can shift from local machine paddlepaddle code to distributed code  **within ten lines of code**. Second, different algorithms can be easily defined through **distributed strategy**  through Fleet API. Finally, distributed training is **extremely fast** with Fleet and just enjoy it.

**Note: all the examples here should be replicated from develop branch of Paddle**

## Fleet is Highly Efficient

Deep neural networks training with Fleet API is highly efficient in PaddlePaddle. We benchmark serveral standard models here.

### Parameter Server Training

Parameter server training benchmark is performed on click through rate estimation task on [Criteo Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/data) and Semantic Representation Learning on [One-billion word Dataset](https://ai.google/research/pubs/pub41880). Details of hardware and software information for this benchmark can be found in [parameter server benchmark](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/ps).

<p align="center">
<img align="center" src="../../images/fleet_ps_benchmark_refine.png" height="270px" width="940px">
<p>
    
### Collective Training

Collective Training is usually used in GPU training in PaddlePaddle. Benchmark of collective training with Fleet is as follows. Details of hardware and software information for this benchmark can be found in [benchmark environment](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/collective).

<p align="center">
<img src="../../images/fleet_collective_benchmark_refine3.png" height="480px" width="900px">
<p>

## Fleet is Easy To Use

Fleet is easy to use for both collective training and parameter server training. Here is an example for collective training with Fleet.

## More Examples

- [Click Through Estimation](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/ctr)

- [Distribute CTR](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/distribute_ctr)

- [DeepFM](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/deepFM)

- [Semantic Matching](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/simnet_bow)

- [Word2Vec](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/word2vec)

- [Resnet50 on Imagenet](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/collective/resnet)

- [Transformer on En-De](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/collective/transformer)

- [Bert on English Wikipedia](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/collective/bert)

