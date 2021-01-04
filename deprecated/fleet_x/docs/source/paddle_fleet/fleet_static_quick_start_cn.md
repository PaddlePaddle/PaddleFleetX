## 静态图分布式训练快速开始

对于大部分用户来讲，数据并行训练基本可以解决实际业务中的训练要求。我们以一个非常简单的神经网络为例，介绍如何使用飞桨高级分布式API `paddle.distributed.fleet`进行数据并行训练。在数据并行方式下，通常可以采用两种架构进行并行训练，即集合通信训练（Collective Training）和参数服务器训练（Parameter Server Training），接下来的例子会以同样的模型来说明两种架构的数据并行是如何实现的。

### 版本要求

- paddlepaddle-2.0.0-rc-cpu / paddlepaddle-2.0.0-rc-gpu及以上

### 模型描述

为了方便说明，我们采用两层全连接网络的分类模型，并使用`CrossEntropyLoss`来评价模型是否优化的符合目标，数据方面我们采用`Paddle`内置的`Mnist`数据集，存放在`model.py`

``` python

import paddle
import paddle.static.nn as nn

paddle.enable_static()
def mnist_on_mlp_model():
    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    test_dataset = paddle.vision.datasets.MNIST(mode='test')
    x = paddle.data(name="x", shape=[64, 1, 28, 28], dtype='float32')
    y = paddle.data(name="y", shape=[64, 1], dtype='int64')
    x_flatten = paddle.reshape(x, [64, 784])
    fc_1 = nn.fc(input=x_flatten, size=128, act='tanh')
    fc_2 = nn.fc(input=fc_1, size=128, act='tanh')
    prediction = nn.fc(input=[fc_2], size=10, act='softmax')
    cost = paddle.fluid.layers.cross_entropy(input=prediction, label=y)
    acc_top1 = paddle.fluid.layers.accuracy(input=prediction, label=y, k=1)
    avg_cost = paddle.fluid.layers.mean(x=cost)
    return train_dataset, test_dataset, x, y, avg_cost, acc_top1

    
```

### 采用GPU多机多卡进行同步训练

`collective_trainer.py`

``` python
import os
import paddle
import paddle.distributed.fleet as fleet
from model import mnist_on_mlp_model

train_data, test_data, x, y, cost, acc = mnist_on_mlp_model()
place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
train_dataloader = paddle.io.DataLoader(
    train_data, feed_list=[x, y], drop_last=True,
    places=place, batch_size=64, shuffle=True)
fleet.init(is_collective=True)
strategy = fleet.DistributedStrategy()
#optimizer = paddle.optimizer.Adam(learning_rate=0.01)
optimizer = paddle.fluid.optimizer.Adam(learning_rate=0.001)
optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
optimizer.minimize(cost)

exe = paddle.static.Executor(place)
exe.run(paddle.static.default_startup_program())

epoch = 10
step = 0
for i in range(epoch):
    for data in train_dataloader():
        step += 1
        loss_val, acc_val = exe.run(
		  paddle.static.default_main_program(),
		  feed=data, fetch_list=[cost.name, acc.name])
	
```

- 单机四卡训练启动命令
``` shell
fleetrun --gpus 0,1,2,3 collective_trainer.py
```

### 采用参数服务器进行多机训练

`parameter_server_trainer.py`

``` python

import paddle
import paddle.distributed.fleet as fleet
from model import mnist_on_mlp_model

paddle.enable_static()

train_data, test_data, x, y, cost, acc = mnist_on_mlp_model()

fleet.init()
strategy = fleet.DistributedStrategy()
strategy.a_sync = True
optimizer = paddle.fluid.optimizer.Adam(learning_rate=0.001)
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(cost)

if fleet.is_server():
   fleet.init_server()
   fleet.run_server()
else:
   place = paddle.CPUPlace()
   exe = paddle.static.Executor(place)
   exe.run(paddle.static.default_startup_program())
   fleet.init_worker()

   train_dataloader = paddle.io.DataLoader(
      train_data, feed_list=[x, y], drop_last=True, places=place,
      batch_size=64, shuffle=True)

   epoch = 1
   for i in range(epoch):
      for data in train_dataloader():
         cost_val, acc_val = exe.run(
            paddle.static.default_main_program(),
            feed=data, fetch_list=[cost.name, acc.name])
         print("loss: {}, acc: {}".format(cost_val, acc_val))
   fleet.stop_worker()

```

- 两节点Server，两节点Worker的启动命令
``` shell
fleetrun --worker_num 2 --server_num 2 parameter_server_trainer.py
```
