
# Fleet

**Fleet** is High-Level API for distributed training in PaddlePaddle. The name of **Fleet** means that a large crowd of ships working together to finish a large scale job. The design of **Fleet** makes a trade-off between easy-to-use and algorithmic extensibility. First, a user can shift from single machine paddle fluid code to distributed code within ten lines of code. Second, different algorithms can be easily defined through distributed strategy through **Fleet** API.

## Quick Start

We show quick-start examples for user to use **Collective** training with **Fleet API**. Multiple GPU training is frequently used in modern AI models that require high performance computing ability of deep learning framework. Sample codes can be found in examples/quick-start folder.

Suppose we define a simple neural nets in examples/quick-start/nets.py
```python
def mlp(input_x, input_y, hid_dim=128, label_dim=2):
    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim, act='tanh')
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim, act='tanh')
    prediction = fluid.layers.fc(input=[fc_2], size=label_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
    avg_cost = fluid.layers.mean(x=cost)
    return avg_cost
```

Simple local training can be defined as in examples/quick-start/local_trainer.py
```python
import paddle.fluid as fluid
from nets import mlp
from utils import gen_data

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

cost = mlp(input_x, input_y)
optimizer = fluid.optimizer.SGD(learning_rate=0.01)
optimizer.minimize(cost)
place = fluid.CUDAPlace(0)

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
step = 1001
for i in range(step):
    exe.run(feed=gen_data())
```

If you want to use high performance chip to do distributed training, such as distributed GPU training, **Fleet API** will help you by adding less than 10 lines of code, source code of this example is in examples/quick-start/collective_trainer.py

```python
import paddle.fluid as fluid
from utils import gen_data
from nets import mlp
from paddle.fluid.incubate.fleet.collective import fleet
from paddle.fluid.incubate.fleet.base import role_maker

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

cost = mlp(input_x, input_y)
optimizer = fluid.optimizer.SGD(learning_rate=0.01)

role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)
optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(cost)

place = fluid.CUDAPlace(0)

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
step = 1001
for i in range(step):
    exe.run(feed=gen_data())
```

Command for distributed training with multiple process on multiple GPU card is as follows:
```
python -m paddle.distributed.launch collective_trainer.py
```

## Design of Fleet
![Fleet API Overview](fleet_design.png)
**Fleet API** aims to help users run distributed training of deep learning models with easy-to-use API. Also, **Fleet API** maintains the ability to plugin new distributed algorithm by developers of PaddlePaddle.

### Role Maker
A **Role Maker** specifies distributed node role in a distributed training job. For example, in parameter server training scenario, a **Role Maker** appoints current node as a worker or a server, and total node number of current distributed job will be available in **Role Maker**. Currently supported **Role Makers** are as follows:

- **MPISymetricRoleMaker**

  MPISymetricRoleMaker is designed for worker and server assignment
  under MPI. Typically, a worker and a server node will be appointed
  on each physical node. This role maker can be only used under MPI.

- **UserDefinedRoleMaker**

  UserDefinedRoleMaker is designed for worker and server assignment
  under manual. Typically, a worker and a server node will be appointed
  on each physical node, It can be assign by user.
  
- **UserDefinedCollectiveRoleMaker**

  UserDefinedCollectiveRoleMaker is designed for worker assignment
  under manual for collective mode.
  
- **PaddleCloudRoleMaker**

  PaddleCloudRoleMaker is an example case for distributed training job in cloud environment that some environment variables   are predefined. 
  
- **UserDefinedRoleMaker**

  UserDefinedRoleMaker is designed for Parameter Server Training that a user can assign server and worker informance such as server number, trainer number, server endpoints.

### Fleet Mode
A **Fleet** API is available in https://github.com/PaddlePaddle/Paddle, a user can easily import different modes of Fleet APIk. Current available **Fleet Mode** are as follows:
- PSLib Mode: A customized parameter server architecture.
- Distribute Transpiler Mode: Use distributed transpiler to build parameter server training configurations.
- Collective Mode: Use distributed transpiler to build collective training configurations.

#### Distribute Transpiler Mode

```python
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
# your program
role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)

optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(avg_cost)

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
elif fleet.is_worker():
    fleet.init_worker()
    # run program on worker
    fleet.stop_worker()
```

Execution command

```bash
python -m paddle.distributed.launch --worker_num 2 --server_num 2 ps_train.py
```

#### Collective Mode

```python
from paddle.fluid.incubate.fleet.collective import fleet
# your program
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)

optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(avg_cost)

# run your program
```

Execution command

```base
python -m paddle.distributed.launch collective_train.py
```

## Examples

### Transpiler Mode
- [Click Through Estimation](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/ctr)

- Word2Vec

- Semantic Matching

- Resnet50 on Imagenet

- VGG16 on Imagenet

- Bert on English Wikipedia

- Transformer on En-De


## Benchmark

- [Click Through Estimation](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/ps/ctr)

- Word2Vec

- Semantic Matching

- Resnet50 on Imagenet

- VGG16 on Imagenet

- Bert on English Wikipedia

- Transformer on En-De
