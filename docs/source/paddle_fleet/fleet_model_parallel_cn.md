# 1. 基于底层分布式API的模型并行训练

## 1.1 模型并行简介

研究表明，随着模型规模的扩大，往往能够取得更好的任务性能。然而，随着模型采用更深、更宽的网络层，模型的参数规模也随之增长，甚至是超过计算设备的显存或者内存容量。

使用模型并行可以将模型参数放置到多个计算设备，从而降低单个计算设备的显存或者内存消耗，使得大规模神经网络模型的训练成为可能。理论上讲，使用足够多的计算设备可以训练任意规模的模型。

本文档以简单的分类网络为例介绍如何使用飞桨的底层集合通信API实现模型并行训练。

本文档使用的网络结构如下所示，主要包含三层为卷积层和一层全连接层。



![示例模型](img/model_parallel_1.png)



## 1.2 模型并行原理和实现

### 1.2.1 版本要求

* paddlepaddle 2.0-rc-gpu版本及以上

### 1.2.2 实现原理

在分布式训练过程中，综合采用数据并行和模型并行，具体地，卷积层采用数据并行，全连接层采用模型并行，即将全连接层划分到多个计算设备上，每个设备负责各自独立部分地全连接层计算。

![全连接层模型并行示例](img/model_parallel_2.png)

模型并行逻辑简单，简单地将全连接层参数按列切分到多个计算设备上，如上图所示。图例中，M(n)表示将矩阵M按行切分为N块，并取第n个分块；M[N]表示将矩阵M按列切分为N块，并取第n个分块。

具体地讲，各个计算设备分别以各自地样本逐层计算卷积层输出，分别得到最后一层卷积层地输出X(0, X(1) ..., X(N-1)。全连接层计算过程如下：

1. 首先，

```python
# -*- coding: UTF-8 -*-
import paddle
import paddle.nn as nn

# 定义模型并行的全连接网络，需要继承自nn.Layer
class ModelParallelLinear(nn.Layer):
    def __init__(self,
                 in_dim,
                 rank_num,
                 rank_id,
                 class_num):
        super(ModelParallelLinear, self).__init__()
        if class_num % rank_num:
            raise ValueError("Number of classes must be divisible "
                             "the number of ranks.")
        shard_dims = class_num // rank_num
        self.linear = nn.Linear(in_dim, shard_dims)
    
    def forward(self, x):
        global_x_list = []
        paddle.distributed.all_gather(global_x_list, x)
        global_x = paddle.concat(global_x, axis=0)
        out = self.linear(global_x)
        global_out_list = []
        paddle.distributed.all_gather(global_out_list, out)
        all_outs = paddle.concat(global_out_list, axis=1)
        out = paddle.split(global_out_list, rank_num)[rank_id]
        return out
```

```python
# -*- coding: UTF-8 -*-
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
#分布式step 1: 导入paddle.distributed.fleet包
from paddle.distributed import fleet
from model_parallel_linear import ModelParallelLinear

# 定义全连接网络，需继承自nn.Layer
class SimpleModelParallelClassifierNet(nn.Layer):
    def __init__(self,
                 class_num,
                 rank_num,
                 rank_id):
        super(SimpleModelParallelClassifierNet, self).__init__()
        self.conv1 = nn.Conv2D(num_channels=1, num_filters=6, filter_size=5, act='sigmoid')
        self.max_pool1 = nn.Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = nn.Conv2D(num_channels=6, num_filters=16, filter_size=5, act='sigmoid')
        self.max_pool2 = nn.Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv3 = nn.Conv2D(num_channels=16, num_filters=120, filter_size=4, act='sigmoid')
        self.model_parallel_linear = ModelParallelLinear(120,
                                                         rank_num,
                                                         rank_id,
                                                         class_num)
    
    def forward(x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        out = self.model_parallel_linear(x)
        return out

# 分布式step 2: 初始化fleet
fleet.init(is_collective=True)

# 1. 定义网络对象，损失函数和优化器
layer = SimpleModelParallelClassifierNet(class_num=10,
                                         rank_num=fleet.worker_num,
                                         rank_id=fleet.worker_index)
adam = paddle.optimizer.Adam(learning_rate=0.001,
                             parameters=layer.parameters())

# 分布式step 3: 通过fleet获取分布式优化器和分布式模型
adam = fleet.distributed_optimizer(adam)
dp_layer = fleet.distributed_model(layer)


for step in range(20):
    # 2. 执行前向网络
    image = paddle.randn([32, 32], 'float32')
    label = paddle.randint(low=0, high=10)
    output = dp_layer(image)
    loss = F.softmax_with_cross_entropy(output, label)
    loss = paddle.mean(loss)

    print("step:{}\tloss:{}".format(step, loss.numpy()))

    # 3. 执行反向计算和参数更新
    # 分布式step 4: 在执行反向（backward函数）前后进行损失缩放和反向梯度的聚合
    loss = dp_layer.scale_loss(loss)
    loss.backward()
    dp_layer.apply_collective_grads()

    adam.step()
    adam.clear_grad()
```

将上述代码保存为train.py，假设要运行2卡任务，那么只需要在命令行执行下面的命令：

```shell
fleetrun --gpus=0,1 tain.py
```

