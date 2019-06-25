
# Fleet

**Fleet** is High-Level API for distributed training in PaddlePaddle. The name of **Fleet** means that a large crowd of ships working together to finish a large scale job. The design of **Fleet** makes a trade-off between easy-to-use and algorithmic extensibility. First, a user can shift from single machine paddle fluid code to distributed code within ten lines of code. Second, different algorithms can be easily defined through distributed strategy through **Fleet** API.

## Quick Start
```
import incubator as incubate
```

## Design of Fleet
![Fleet API Overview](fleet_design.png)

### Role Maker
A **Role Maker** specifies distributed node role in a distributed training job. For example, in parameter server training scenario, a **Role Maker** appoints current node as a worker or a server, and total node number of current distributed job will be available in **Role Maker**. Currently supported **Role Makers** are as follows:
- MPISymetricRoleMaker
- UserDefinedRoleMaker
- PaddleCloudRoleMaker

### Fleet Mode
A **Fleet** API is available in https://github.com/PaddlePaddle/Paddle, a user can easily import different modes of Fleet APIk. Current available **Fleet Mode** are as follows:
- PSLib Mode
- Distribute Transpiler Mode
- Collective Mode

#### PSLib Mode
```
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet

```

#### Distribute Transpiler Mode
```
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
# 该模型运行在单个CPU上
place = fluid.CPUPlace()

role = PaddleCloudRoleMaker()
fleet.init(role)
# 调用train_program 获取预测值，损失值，
prediction, [avg_loss, acc] = train_program()

# 输入的原始图像数据，大小为28*28*1
img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
# 标签层，名称为label,对应输入图片的类别标签
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
# 告知网络传入的数据分为两部分，第一部分是img值，第二部分是label值
feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

# 选择Adam优化器
optimizer = fluid.optimizer.Adagrad(learning_rate=0.001)
optimizer = fleet.distribute_optimizer(optimizer)
optimizer.minimize(avg_loss)

if fleet.is_server():
  # start server
else:
  # do training
  
```

#### Collective Mode
```
from paddle.fluid.incubate.fleet.collective import fleet

```
run training

## Benchmark

- Click Through Estimation

- Word2Vec

- Semantic Matching

- Resnet50 on Imagenet

- VGG16 on Imagenet

- Bert on English Wikipedia

- Transformer on En-De
