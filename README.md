
# Fleet

**Fleet** is High-Level API for distributed training in PaddlePaddle. The name of **Fleet** means that a large crowd of ships working together to finish a large scale job. The design of Fleet makes a trade-off between easy-to-use and algorithmic extensibility and is highly efficient. First, a user can shift from local machine paddlepaddle code to distributed code  **within ten lines of code**. Second, different algorithms can be easily defined through **distributed strategy**  through Fleet API. Finally, distributed training is **extremely fast** with Fleet and just enjoy it.

**Note: all the examples here should be replicated from develop branch of Paddle**

## Installation of Fleet-Lightning
To show how to setup distributed training with fleet, we introduce a small library call **fleet-lightning**. **fleet-lightning** helps industrial users to directly train a specific standard model such as Resnet50 without learning to write a Paddle Model. 

``` bash
pip install fleet-x
```

## A Distributed Resnet50 Training Example

``` python
import os
import paddle
import paddle.distributed.fleet as fleet
import fleetx as X

configs = X.parse_train_configs()

fleet.init(is_collective=True)
model = X.applications.Resnet50()
loader = model.load_imagenet_from_file("/pathto/imagenet/train.txt")

optimizer = paddle.optimizer.Momentum(learning_rate=configs.lr, momentum=configs.momentum)
optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer)
optimizer.minimize(model.loss)

```
