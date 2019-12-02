# DGC示例
DGC代码示例修改自[paddle book 数字识别](https://github.com/PaddlePaddle/book/tree/develop/02.recognize_digits)代码, 本示例主要用于展示DGC的使用方式。代码不定时更新，如需获取最新数字识别源码及相关教程，请查阅其[源代码库](https://github.com/PaddlePaddle/book/tree/develop/01.recognize_digits)及[相应文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basics/recognize_digits/README.cn.html)。

### 说明 ###
1. 硬件环境要求：
DGC目前只支持GPU多卡及分布式collective训练，需要有相应的cuda、cuDNN、nccl环境。
2. Paddle环境要求：
DGC只支持GPU，所以需GPU版本的Paddle。DGC依赖Paddle Fluid 1.6.2及以上版本或最新develop分支版本，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装。
3. python脚本说明：
该目录下有[train.py](./train.py)和[train_with_fleet.py](./train_with_fleet.py)两个python脚本。其中train.py是原代码库中的训练脚本，只支持单机单卡模式；由于DGC需要多卡或分布式环境运行，所以对train.py脚本修改添加fleet分布式接口及DGC代码形成train_with_fleet.py训练脚本，同时保留train.py脚本以方便用户对比。
4. DGC注意事项： 
现有fuse策略会造成DGC失效，所以使用DGC务必关闭fuse，目前train_with_fleet.py脚本中已添加相应逻辑。同时DGC只支持Momentum，所以对于Adam等优化器需迁移为Momentum才能进一步迁移为DGC。

### 如何运行DGC ###
以单机多卡为例，配置好相应的cuda运行环境，执行
``` bash
sh run_dgc.sh
```
