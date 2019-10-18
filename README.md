
# Fleet

**Fleet** is High-Level API for distributed training in PaddlePaddle. The name of **Fleet** means that a large crowd of ships working together to finish a large scale job. The design of **Fleet** makes a trade-off between easy-to-use and algorithmic extensibility. First, a user can shift from single machine paddle fluid code to distributed code within ten lines of code. Second, different algorithms can be easily defined through distributed strategy through **Fleet** API.

## Fleet is Highly Efficient

Deep neural networks training with Fleet API is highly efficient in PaddlePaddle. We benchmark serveral standard models here.

### Parameter Server Training

Parameter server training benchmark is performed on click through rate estimation task on [Criteo Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/data). Details of hardware and software information for this benchmark can be found in .

<p align="center">
<img align="center" src="images/ctr.png" height="250px" width="470px">
<p>
    
### Collective Training

Collective Training is usually used in GPU training in PaddlePaddle. Benchmark of collective training with Fleet is as follows. Details of hardware and software information for this benchmark can be found in .

<p align="center">
<img src="images/joint_benchmark.png" height="480px" width="850px">
<p>

## Easy To Use

Fleet is easy to use for both collective training and parameter server training. Here is an example for collective training with Fleet.


![Fleet API Overview](fleet_design.png)

# Fleet

**Fleet** is High-Level API for distributed training in PaddlePaddle. The name of **Fleet** means that a large crowd of ships working together to finish a large scale job. The design of **Fleet** makes a trade-off between easy-to-use and algorithmic extensibility. First, a user can shift from single machine paddle fluid code to distributed code within ten lines of code. Second, different algorithms can be easily defined through distributed strategy through **Fleet** API.

## Quick Start

**Note: all the examples here should be replicated from develop branch of Paddle**

We show quick-start examples for user to use **Collective** training with **Fleet API**. Multiple GPU training is frequently used in modern AI models that require high performance computing ability of deep learning framework. Sample codes can be found in examples/quick-start folder.

Suppose we define a simple neural nets in examples/quick-start/nets.py
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
    cost_val = exe.run(program=fluid.default_main_program(),
                       feed=gen_data(),
                       fetch_list=[cost.name])
    print("step%d cost=%f" % (i, cost_val[0]))
```


Parameter server training is suitable for parallel training of large-scale data. Based on the definition of single-machine model, we give an example of training with parameter server, source code of this example is in examples/quick-start/distributed_train.py
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

Command for parameter server training with multiple process on CPU is as follows:
```
python -m paddle.distributed.launch_ps distributed_train.py
```

If you want to use high performance chip to do distributed training, such as distributed GPU training, **Fleet API** will help you by adding less than 10 lines of code, source code of this example is in examples/quick-start/collective_train.py

```python
import paddle.fluid as fluid
from nets import mlp
from paddle.fluid.incubate.fleet.collective import fleet
from paddle.fluid.incubate.fleet.base import role_maker
from utils import gen_data

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

cost = mlp(input_x, input_y)
optimizer = fluid.optimizer.SGD(learning_rate=0.01)
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)

optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(cost)
place = fluid.CUDAPlace(0)

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
step = 1001
for i in range(step):
    cost_val = exe.run(program=fluid.main_program,
                       feed=gen_data(),
                       fetch_list=[cost.name])
    print("worker_index: %d, step%d cost = %f" % 
          (fleet.worker_index(), i, cost_val[0]))
```

Command for distributed training with multiple process on multiple GPU card is as follows:
```
python -m paddle.distributed.launch collective_train.py
```

## Design of Fleet

![Fleet API Overview](fleet_design.png)
**Fleet API** aims to help users run distributed training of deep learning models with easy-to-use API. Also, **Fleet API** maintains the ability to plugin new distributed algorithm by developers of PaddlePaddle.

### Fleet API functions


- **init(role_maker=None)**

  It is responsible for fleet initialization, which needs to be called before using other Fleet interfaces. Basically it is used to define multi-machine environment configuration.   

- **is_worker()**

  It is used to check whether the current node is a worker node in parameter server training. If the current node is a worker node,the function will return True, otherwise the function will return False.

- **is_server(model_dir=None)**

  It is used to check whether the current node is a server node in parameter server training. If the current node is a server node,the function will return True, otherwise the function will return False.

- **init_server()**

  Fleet loads model-related parameters stored in model_dir to initialize parameter server in parameter server training.

- **run_server()**

  It is used to start the server service in parameter server training.

- **init_worker()**

  It is used to start the worker service in parameter server training.

- **stop_worker()**

  It is used to terminate the worker service after the parameter server training is done.

- **distributed_optimizer(optimizer, strategy=None)**

  It is used to  decorate the distributed optimization algorithm. Users can bring in single optimizer and configure distributed training strategy, the function will return a distributed optimizer

### Role Maker

- **MPISymetricRoleMaker**

**description**：MPISymetric RoleMaker assumes that each node starts two processes, one for worker service and one for parameter server service, which requires an MPI environment on the user's cluster.

**example**：

```python
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.base import role_maker

role = role_maker.MPISymetricRoleMaker()
fleet.init(role)
```
**command for running the example**：

```base
mpirun -np 2 python trainer.py
```

- **PaddleCloudRoleMaker**

**description**：Paddle CloudRoleMaker is a high-level encapsulation that supports starting scripts using paddle.distributed.launch or paddle.distributed.launch_ps.

**Parameter Server training example**：
```python
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.base import role_maker

role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)
```
**command for running the example**：
```
python -m paddle.distributed.launch_ps --worker_num 2 --server_num 2 trainer.py
```
**Collective training example**：
```python
from paddle.fluid.incubate.fleet.collective import fleet
from paddle.fluid.incubate.fleet.base import role_maker

role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)
```
**command for running the example**：
```
python -m paddle.distributed.launch --selected_gpus="0,1,2,3" trainer.py
```
- **UserDefinedRoleMaker**

**description**：User-defined node role information, IP and port information.

**example**：
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

### Strategy
**Parameter Server Training**
- Sync_mode

**Collective Training**
- LocalSGD
- ReduceGrad

### Fleet Mode

#### Parameter Server Training

```python
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
```

#### Collective Training

```python
from paddle.fluid.incubate.fleet.collective import fleet
```

## Examples

- [Click Through Estimation](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/ctr)

- [Distribute CTR](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/distribute_ctr)

- [DeepFM](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/deepFM)

- [Semantic Matching](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/simnet_bow)

- [Word2Vec](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/word2vec)

- [Resnet50 on Imagenet](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/resnet50)

- [Transformer on En-De](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/transformer)

- [Bert on English Wikipedia](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/bert)


