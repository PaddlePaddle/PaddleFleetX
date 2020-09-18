## 快速开始
以下通过图像分类Resnet50的例子，说明如何使用FleetX的接口进行分布式训练。具体步骤如下：

1. 导入依赖
2. 构建模型
3. 定义分布式策略
4. 开始训练

下面章节会对每个步骤进行讲解。

### 1. 导入依赖

FleetX依赖Paddle 1.8.0及之后的版本。请确认已安装正确的Paddle版本，并按照以下方式导入Paddle 及 FleetX。
```
import paddle
import paddle.distributed.fleet as fleet
import fleetx as X
```


### 2. 构建模型

通过FleetX提供的 `X.applications` 接口，用户可以使用一行代码加载一些经典的深度模型，如：Resnet50，VGG16，BERT，Transformer等。同时，用户可以使用一行代码加载特定格式的数据，如对于图像分类任务，用户可以加载ImageNet格式的数据。


```
# configs中记录了训练相关的参数，如学习率等
configs = X.parse_train_configs()
# 加载模型至model，并加载数据至dataloader
model = X.applications.Resnet50()
loader = model.load_imagenet_from_file("/pathto/ImageNet/train.txt")

# 定义优化器
optimizer = paddle.optimizer.Momentum(learning_rate=configs.lr, momentum=configs.momentum)
```

### 3. 定义分布式策略

在定义完单机训练网络后，用户可以使用`paddle.distributed.fleet.DistributedStrategy()`接口定义分布式策略，将模型转成分布式模式。

```
# 使用collective GPU分布式模式
fleet.init(is_collective=True)
# 使用默认的分布式策略
dist_strategy = ifleet.DistributedStrategy()
# 将单机模型转换为分布式
optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)
```

### 4. 开始训练

最后用户可以开始训练定义好的模型。

```
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
