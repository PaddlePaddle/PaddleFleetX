## FleetX快速开始

### FleetX是什么？
`FleetX`提供效率最高的分布式模型预训练功能，它可以作为`paddle.distributed.fleet`的扩展进行配合使用。

### 提供哪些功能？
- 短代码定义预训练模型
- 预置经典模型的公开训练数据
- 用户可低成本替换自有数据集
- 面向每个模型的最佳分布式训练实践

### 上手示例
以下通过图像分类Resnet50的例子，说明如何使用FleetX的接口进行分布式训练。具体步骤如下：

- 导入依赖
- 构建模型
- 定义分布式策略
- 开始训练

为了简化模型定义的过程，我们在后面的文档中会尽量使用FleetX封装的高级API，方便用户理解分布式训练的核心内容。

#### 1. 导入依赖

FleetX依赖Paddle 1.8.0及之后的版本。请确认已安装正确的Paddle版本，并按照以下方式导入Paddle 及 FleetX。
``` python
import paddle
import paddle.distributed.fleet as fleet
import fleetx as X
```

#### 2. 构建模型

通过FleetX提供的 `X.applications` 接口，用户可以使用一行代码加载一些经典的深度模型，如：Resnet50，VGG16，BERT，Transformer等。同时，用户可以使用一行代码加载特定格式的数据，如对于图像分类任务，用户可以加载ImageNet格式的数据。

``` python
import paddle
import paddle.distributed.fleet as fleet
import fleetx as X

configs = X.parse_train_configs()

model = X.applications.Resnet50()
downloader = X.utils.Downloader()
local_path = downloader.download_from_bos(fs_yml="https://xxx.xx.xx.xx/full_imagenet_bos.yml", local_path='./data')
loader = model.get_train_dataloader(local_path, batch_size=32)

```

#### 3. 定义分布式策略

在定义完单机训练网络后，用户可以使用`paddle.distributed.fleet.DistributedStrategy()`接口定义分布式策略，将模型转成分布式模式。

``` python
# 使用paddle.distributed.fleet进行collective training
fleet.init(is_collective=True)
# 定义DistributedStrategy
dist_strategy = fleet.DistributedStrategy()
# 装饰单机optimizer为分布式optimizer
optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)
```

#### 4. 开始训练

可以使用`FleetX`内置的训练器进行快速训练，方便算法工程师快速上手：

``` python
trainer = X.MultiGPUTrainer()
trainer.fit(model, loader, epoch=10)
```

用户也可以采用Paddle原生的API进行训练流程的定义，代码如下：

``` python
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

for epoch_id in range(5):
    step_id = 0
    for data in loader:
        cost_val = exe.run(paddle.default_main_program(),
		           feed=data,
		           fetch_list=[model.loss.name])
        if step_id % 100 == 0:
            print("worker index: %d, epoch: %d, step: %d, train loss: %f"
                 % (fleet.worker_index(), epoch_id, step_id, cost_val[0]))
```	

从Paddle 2.0 rc版本开始，我们统一采用fleetrun进行多卡训练的启动，方式如下：

``` shell
fleetrun --gpus 0,1,2,3 resnet_app.py
```

关于`fleetrun`命令，更详细的使用说明请参考[fleetrun](fleetrun_usage_cn.html)