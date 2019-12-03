# FleetAPI
Fleet是PaddlePaddle Fluid最新优化的多机High-Level API， 统一了多机API的实现，兼容Transpiler/Collective两种模式。 可以在MPI、K8S、PaddleCloud以及用户自定义环境下进行多机训练，以及自定义分布式训练配置，Fleet的设计在易用性和算法可扩展性方面做出了权衡。
使用FleetAPI， 用户可以从如下几个方面获得收益：
- 添加少量代码即可从单机训练切换到大规模分布式训练
- 可以使用多种针对分布式训练的算法优化及图优化
- 极大的性能提升及训练规模的提升

## API介绍
目前FleetAPI针对CPU分布训练相关的API有10个，这10个API涵盖了目前分布式训练的全部生命周期。 具体API说明如下：

### Fleet
--------

fleet.init(role_maker=None)

fleet初始化，需要在使用fleet其他接口前先调用，用于定义多机的环境配置。

- 参数:
    + role_maker(RoleMakerBase|None) — 多机环境配置，目前有MPISymetricRoleMaker(默认)和UserDefinedRoleMaker、PaddleCloudRoleMaker等多种。
    + MPISymetricRoleMaker在MPI集群下使用，需要自定义环境变量可使用UserDefinedRoleMaker。

- 返回类型: None

- 代码示例：
``` python
exe = fluid.Executor(fluid.CPUPlace())
role = UserDefinedRoleMaker(current_id=0,
                 role=Role.WORKER,
                 worker_num=3,
                 server_endpoints=["127.0.0.1:6001","127.0.0.1:6002"])
fleet.init(role_maker=role)
```

--------

fleet.distributed_optimizer(optimizer, strategy=None)

分布式优化算法装饰器，用户可带入单机optimizer，并配置分布式训练策略，返回一个分布式的optimizer

- 参数：
    + optimizer (Optimizer) — 当前网络定义的优化器SGD/ADAM等。
    + strategy(Any|None) — 多机策略配置，根据fleet的实现自行配置，DistributedTranspiler和Collective模式指定为DistributeTranspilerConfig。

- 返回类型：DistributedOptimizer

- 代码示例：
``` python
optimizer = fluid.optimizer.SGD(learning_rate=0.1)
config = DistributeTranspilerConfig()
config.sync_mode = False
optimizer = fleet.distributed_optimizer(optimizer, config)
optimizer.minimize(cost)
```

--------

fleet.is_server()

判断当前节点是否是Server节点， 是则返回True，否则返回False。在CPU分布式训练下， 节点类型分为trainer和pserver两类。

- 参数： None
- 返回类型: bool

- 代码示例：
``` python
if fleet.is_server():
    fleet.run_server()
```

--------

fleet.init_server(model_dir=None)

加载model_dir中保存的模型相关参数进行PServer的初始化

- 参数:
    + model_dir (str|None) — 模型参数保存的目录。模型参数来自于fleet.save_persistable或者fluid.io.save_persistable保存下来的参数。

- 返回类型: None

- 代码示例：
``` python
if fleet.is_server():
    model_dir = "xxx"
    fleet.init_server(model_dir)
    fleet.run_server()
```

--------

fleet.run_server()

启动PServer的进程， 此进程为常驻进程， 会一直监听来自trainer端的消息。 当前版本不会主动退出。

- 参数: None
- 返回类型: None

- 代码示例：
``` python
if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
```

--------

fleet.is_worker()

判断当前节点是否是Worker节点， trainer会启动训练。
- 参数: None
- 返回类型: None

--------

fleet.init_worker()

如果是worker节点，则会根据当前启动的模式进行针对性的初始化。
- 参数: None
- 返回类型: None

--------

fleet.save_inference_model

CPU分布式专用的模型保存接口，在trainer端调用，根据用户配置保存模型参数及模型文件， 具体用法参考[save_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/io_cn/save_inference_model_cn.html#save-inference-model) 。

--------

fleet.save_persistable

CPU分布式专用的模型保存接口，在trainer端调用，根据网络保存完整的模型参数， 具体用法参考[save_persistables](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/io_cn/save_persistables_cn.html#save-persistables)。

--------

### RoleMaker

--------

MPISymetricRoleMaker

MPISymetricRoleMaker会假设每个节点启动两个进程，1worker+1pserver，这种RoleMaker要求用户的集群上有mpi环境。

- 示例：
```python
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.base import role_maker

role = role_maker.MPISymetricRoleMaker()
fleet.init(role)
```

--------

PaddleCloudRoleMaker

PaddleCloudRoleMaker是一个高级封装，支持使用paddle.distributed.launch_ps启动脚本

- 示例：
```python
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.base import role_maker

role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)
```

--------

UserDefinedRoleMaker

用户自定义节点的角色信息，IP和端口信息

- 示例：
```python
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.base import role_maker

role = role_maker.UserDefinedRoleMaker(
            current_id=int(os.getenv("CURRENT_ID")),
            role=role_maker.Role.WORKER if bool(int(os.getenv("IS_WORKER")))
                                                                            else role_maker.Role.SERVER,
            worker_num=int(os.getenv("WORKER_NUM")),
            server_endpoints=pserver_endpoints)
fleet.init(role)
```
--------


## 使用说明
Fleet代码位于python/paddle/fluid/incubate/fleet下， 对外的实例名为fleet， 使用Transpiler模式， 请使用：
```
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
```

## 运行流程及原理

## Fleet API快速上手示例
下面会针对Fleet API最常见的两种使用场景，用一个模型做示例，目的是让用户有快速上手体验的模板。快速上手的示例源代码可以在 [Fleet Quick Start] (https://github.com/PaddlePaddle/Fleet/tree/develop/examples/quick-start) 找到。

假设我们定义MLP网络如下：
```python
import paddle.fluid as fluid

def mlp(input_x, input_y, hid_dim=128, label_dim=2):
  fc_1 = fluid.layers.fc(input=input_x, size=hid_dim, act='tanh')
  fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim, act='tanh')
  prediction = fluid.layers.fc(input=[fc_2], size=label_dim, act='softmax')
  cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
  avg_cost = fluid.layers.mean(x=cost)
  return avg_cost
```

定义一个在内存生成数据的Reader如下：
```python
import numpy as np

def gen_data():
    return {"x": np.random.random(size=(128, 32)).astype('float32'),
            "y": np.random.randint(2, size=(128, 1)).astype('int64')}
```

单机Trainer定义:
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
  cost_val = exe.run(feed=gen_data(), fetch_list=[cost.name])
  print("step%d cost=%f" % (i, cost_val[0]))
```

基于Parameter Server的CPU分布训练方法：
```python
import paddle.fluid as fluid
from nets import mlp
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.base import role_maker
from utils import gen_data

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

cost = mlp(input_x, input_y)
optimizer = fluid.optimizer.SGD(learning_rate=0.01)

role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)
optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(cost)

if fleet.is_server():
  fleet.init_server()
  fleet.run_server()
elif fleet.is_worker():
  place = fluid.CPUPlace()
  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())
  step = 1001
  for i in range(step):
    cost_val = exe.run(
        program=fluid.default_main_program(),
        feed=gen_data(),
        fetch_list=[cost.name])
    print("worker_index: %d, step%d cost = %f" %
         (fleet.worker_index(), i, cost_val[0]))
```

## TranspilerAPI
TranspilerAPI是老版本的分布式训练API，在设计和实现上有诸多不足之处，如果是新用户，请尽可能选择FleetAPI。Transpiler API可以把单机可以执行的程序快速转变成可以分布式执行的程序。在不同的服务器节点 上，通过传给 transpiler 对应的参数，以获取当前节点需要执行的 Program。

需要配置参数包括：

| 参数     |  说明
| ------------- | ------------- |
| role     | 区分作为pserver启动还是trainer启动，不传给transpile，也可以用其他的变量名或环境变量
| trainer_id   |  如果是trainer进程，用于指定当前trainer在任务中的唯一id，从0开始，在一个任务中需保证不重复
| pservers   |   当前任务所有pserver的ip:port列表字符串，形式比如：127.0.0.1:6170,127.0.0.1:6171
| trainers  |   trainer节点的个数
| sync_mode |  True为同步模式，False为异步模式，说明(目前分布式的其他模型需要配置此参数联合其他配置完成)
| startup_program | 如果startup_program不是默认的fluid.default_startup_program()，需要传入此参数
| current_endpoint | NCCL2模式需要传这个参数，且在分布式训练增量训练是需要指定此参数


一个例子，假设有两个节点，分别是 192.168.1.1 和 192.168.1.2 ，使用端口6170，启动4个trainer， 则代码可以写成：

``` python
role = "PSERVER"
trainer_id = 0  # get actual trainer id from cluster
pserver_endpoints = "192.168.1.1:6170,192.168.1.2:6170"
current_endpoint = "192.168.1.1:6170" # get actual current endpoint
trainers = 4
t = fluid.DistributeTranspiler()
t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
if role == "PSERVER":
    pserver_prog = t.get_pserver_program(current_endpoint)
    pserver_startup = t.get_startup_program(current_endpoint,
                                            pserver_prog)
    exe.run(pserver_startup)
    exe.run(pserver_prog)
elif role == "TRAINER":
    train_loop(t.get_trainer_program())
```