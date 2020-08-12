
<h2 align="center">FleetX</h2>

**FleetX** is an extension package for [Paddle](https://github.com/PaddlePaddle/Paddle). As cloud service grows rapidly, distributed training of deep learning model will be a user-facing method for daily research. **FleetX** aims to help Paddle users do distributed training with little effort.
<h3 align="center">Out-of-the-box models for large scale dataset</h3>
<h3 align="center">Best practice distributed strategies</h3>
<h3 align="center">Easy-To-Use on Kubernetes cluster</h3>

**Note: all the examples here should be replicated from develop branch of Paddle**

<h2 align="center">Installation</h2>

``` bash
pip install fleet-x
```

<h2 align="center">A Distributed Resnet50 Training Example</h2>

``` python
import os
import paddle
import paddle.distributed.fleet as fleet
import fleetx as X

# fleet-x
configs = X.parse_train_configs()
model = X.applications.Resnet50()
loader = model.load_imagenet_from_file("/pathto/imagenet/train.txt")

# paddle optimizer definition
optimizer = paddle.optimizer.Momentum(learning_rate=configs.lr, momentum=configs.momentum)

# paddle distributed training code here
fleet.init(is_collective=True)
optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer)

optimizer.minimize(model.loss)

```
