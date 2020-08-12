
<h1 align="center">FleetX</h1>

**FleetX** is an extension package for [Paddle](https://github.com/PaddlePaddle/Paddle) High-Level Distributed Training API `paddle.fleet`. As cloud service grows rapidly, distributed training of deep learning model will be a user-facing approach for daily applications and research. **FleetX** aims to help Paddle users do distributed training on cloud like running on notebooks.

<h2 align="center">Main Features</h2>

- Out of box models suitable for distributed training
- Best practice distributed strategies
- Easy-To-Use Submission, Profiling and Job Monitor

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
