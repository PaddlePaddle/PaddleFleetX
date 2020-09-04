
<h1 align="center">FleetX</h1>

<p align="center"> Fully utilize your GPU Clusters with FleetX for your model pre-training. </p>

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
import fleetx as X
import paddle
import paddle.distributed.fleet as fleet

configs = X.parse_train_configs()

model = X.applications.Resnet50()
imagenet_downloader = X.utils.ImageNetDownloader()
local_path = imagenet_downloader.download_from_bos(local_path='./data')
loader = model.load_imagenet_from_file(
    "{}/train.txt".format(local_path), batch_size=32)

fleet.init(is_collective=True)
dist_strategy = fleet.DistributedStrategy()
dist_strategy.amp = True

optimizer = paddle.optimizer.Momentum(
    learning_rate=configs.lr,
    momentum=configs.momentum,
    weight_decay=paddle.fluid.regularizer.L2Decay(0.0001))
optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
optimizer.minimize(model.loss)

trainer = X.MultiGPUTrainer()
trainer.fit(model, loader, epoch=10)

```

<h2 align="center">How to launch your task</h2>

- Multiple cards

``` shell
fleetrun --gpus 0,1,2,3,4,5,6,7 resnet50_app.py
```
