
<h1 align="center">FleetX</h1>

<p align="center"> Fully utilize your GPUs or other AI Chips with FleetX for your model pre-training. </p>

<h2 align="center">What is it?</h2>

- **FleetX** is an out-of-the-box pre-trained model training toolkit for cloud users. It can be viewed as an extension package for `Paddle's` High-Level Distributed Training API `paddle.distributed.fleet`. 

<h2 align="center">Key Features</h2>

- **Pre-defined Models for Training**: define a Bert-Large or GPT-2 with one line code, which is commonly used self-supervised training model.
- **Friendly to User-defined Dataset**: plugin user-defined dataset and do training without much effort.
- **Distributed Training Best Practices**: the most efficient way to do distributed training is provided.

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
optimizer = fleet.distributed_optimizer(optimizer)

optimizer.minimize(model.loss)
exe = paddle.Executor(paddle.CUDAPlace(0))

epoch = 10
for e in range(epoch):
    for data in loader():
        cost_val = exe.run(paddle.static.default_main_program(), feed=data, fetch_list=[model.loss.name])

```


<h2 align="center">How to launch your task</h2>

- Multiple cards

``` shell
fleetrun --gpus 0,1,2,3,4,5,6,7 resnet50_app.py
```

- Run on Baidu Cloud

``` shell
fleetrun --conf config.yml resnet50_app.py
```


<h2 align="center">Multi-slot DNN CTR model</h2>

``` python
import os
import paddle
import paddle.distributed.fleet as fleet
import fleetx as X

# fleet-x
configs = X.parse_train_configs()
model = X.applications.MultiSlotCTR()
loader = model.load_multislot_from_file("/pathto/imagenet/train.txt")

# paddle optimizer definition
optimizer = paddle.optimizer.SGD(learning_rate=configs.lr)

# paddle distributed training code here
fleet.init()
optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(model.loss)

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
else:
    fleet.init_worker()
    exe = paddle.Executor(paddle.CPUPlace())
    exe.run(paddle.default_startup_program())
    epoch = 10
    for e in range(epoch):
        for data in loader():
            cost_val = exe.run(paddle.default_main_program(), feed=data, fetch_list=[model.loss.name])
    fleet.stop_worker()

```

