
# Fleet

**Fleet** is High-Level API for distributed training in PaddlePaddle. The name of **Fleet** means that a large crowd of ships working together to finish a large scale job. The design of Fleet makes a trade-off between easy-to-use and algorithmic extensibility and is highly efficient. First, a user can shift from local machine paddlepaddle code to distributed code  **within ten lines of code**. Second, different algorithms can be easily defined through **distributed strategy**  through Fleet API. Finally, distributed training is **extremely fast** with Fleet and just enjoy it.

**Note: all the examples here should be replicated from develop branch of Paddle**

## Fleet is Highly Efficient

Deep neural networks training with Fleet API is highly efficient in PaddlePaddle. We benchmark serveral standard models here.

### Parameter Server Training

Parameter server training benchmark is performed on click through rate estimation task on [Criteo Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/data) and Semantic Representation Learning on [One-billion word Dataset](https://ai.google/research/pubs/pub41880). Details of hardware and software information for this benchmark can be found in [parameter server benchmark](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/ps).

<p align="center">
<img align="center" src="images/fleet_ps_benchmark_refine.png" height="270px" width="940px">
<p>

### Collective Training

Collective Training is usually used in GPU training in PaddlePaddle. Benchmark of collective training with Fleet is as follows. Details of hardware and software information for this benchmark can be found in [benchmark environment](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/collective).

<p align="center">
<img src="images/fleet_collective_benchmark_refine3.png" height="480px" width="900px">
<p>

### Mixed precision accelerated collective training throughput

<p align="center">
<img style="float: left;" src="images/fleet_collective_mixed_precision_training.png" height="280px" width="450px">
<p>
<br><br><br><br><br><br><br><br><br><br><br><br><br>

## Fleet is Easy To Use

Fleet is easy to use for both collective training and parameter server training. Here is an example for collective training with Fleet.

Local Single GPU Cards Training

``` python
import paddle.fluid as fluid
from utils import gen_data
from nets import mlp

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

cost = mlp(input_x, input_y)
optimizer = fluid.optimizer.SGD(learning_rate=0.01)
optimizer.minimize(cost, fluid.default_startup_program())

train_prog = fluid.default_main_program()
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

step = 1001
for i in range(step):
    cost_val = exe.run(program=train_prog, feed=gen_data(), fetch_list=[cost.name])
```

Local Multiple GPU Cards Training
``` python
import paddle.fluid as fluid
from utils import gen_data
from nets import mlp
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy  # new line 1 
from paddle.fluid.incubate.fleet.base import role_maker # new line 2

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

cost = mlp(input_x, input_y)
optimizer = fluid.optimizer.SGD(learning_rate=0.01)

role = role_maker.PaddleCloudRoleMaker(is_collective=True) # new line 3
fleet.init(role) # new line 4

optimizer = fleet.distributed_optimizer(optimizer, strategy=DistributedStrategy()) # new line 5
optimizer.minimize(cost, fluid.default_startup_program())

train_prog = fleet.main_program # change line 1
place = fluid.CUDAPlace(0)

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

step = 1001
for i in range(step):
    cost_val = exe.run(program=train_prog, feed=gen_data(), fetch_list=[cost.name])
```

Launch command:
```
python -m paddle.distributed.launch --selected_gpus="0,1,2,3" trainer.py
```

## More Examples

- [Click Through Estimation](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/ctr)

- [Distribute CTR](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/distribute_ctr)

- [DeepFM](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/deepFM)

- [Semantic Matching](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/simnet_bow)

- [Word2Vec](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/word2vec)

- [Resnet50 on Imagenet](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/collective/resnet)

- [Transformer on En-De](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/collective/transformer)

- [Bert on English Wikipedia](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/collective/bert)

