# 使用Local SGD 优化分布式超大batch 训练

## 简介 
在使用 distributed SGD 进行数据并行的分布式训练时，常会遇到以下两个问题：

* 分布式训练的吞吐会受到集群中慢节点（straggling node）和随机通信延迟的影响。
* 数据并行分布式增大了训练实际的batch size，过大的batch size 会影响最终的训练精度。

local SGD 通过延长节点间同步的间隔(局部异步训练)来减轻慢节点的影响和减少通信频率，以此提升训练的吞吐速度；另一方面，为了减小相对于本地训练（小batch size）的精度损失，[DON’T USE LARGE MINI-BATCHES, USE LOCAL SGD](https://arxiv.org/abs/1808.07217) 和 [ADAPTIVE COMMUNICATION STRATEGIES TO ACHIEVE THE BEST ERROR-RUNTIME TRADE-OFF IN LOCAL-UPDATE SGD](https://arxiv.org/abs/1810.08313) 分别提出了：`post-local SGD` 和 `自适应步长 (Adaptive Communication)` 策略，来减少参数同步频率降低带来的精度损失。

<p align="center">
<img src="https://d3i71xaburhd42.cloudfront.net/478dca8410e0e08d2d1010376f4e5e1125ba7909/3-Figure2-1.png" width="250"/>
</p>

在local SGD 训练中，集群中的每个 worker 各自会独立的进行 H 个连续的 SGD 更新， 然后集群中的所有 worker 会进行通信，同步（averaging）所有 workers 上的参数。一个双 workers，同步间隙为3 iterations 的local SGD过程如上图所示。黄绿两条路径表示两个 workers 各自的 local SGD 更新过程，中间的蓝色路径表示同步后的模型所在的位置。

local SGD中的一个关键问题是如何确定参数同步的间隔(频率)，以到达训练吞吐和训练精度间更好的平衡：

* 增大参数同步的间隔可以减少 workers 间通信延迟的影响提高训练吞吐.
* 但增大同步间隔可能会造成最终训练精度的损失。 [[1]](https://arxiv.org/abs/1708.01012)

post-local SGD 将训练过程分成两个阶段：第一阶段 wokers 间同步的间隔为 1 iteration，即同步SGD，来保证最终训练精度；在第二阶段增大同步间隔到固定常数 H iterations，来提升训练吞吐。其公式如下：

Adaptive Communication local SGD 通过动态的调整参数同步的间隔来尝试达到训练吞吐和精度间的更好的平衡。在训练初始或者上一段参数同步完成后，根据如下公式计算一下次参数同步的间隔（iteration）。详细的公式推导和参数定义请参考[原论文](https://arxiv.org/abs/1808.07217)。 

Fleet 中实现了 `Naive local SGD` 和 `Adaptive Communication local SGD` 两种策略。 中下文将给出 Fleet中 local SGD 的实践效果，并通过一个简单例子介绍如何在Fleet 中使用 local SGD。

## Fleet 效果
试验设置

|model|dataset|local batch size|cluster|dtype|warming up| learning rate decay|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|resnet50|Imagenet|128|4 x 8 x V100|FP32|30|polynomial |

试验结果

|local step|qps|acc1|acc5|
|:---:|:---:|:---:|:---:|
|1|	8270.91|0.7579|	0.9266|
|2|	8715.67|0.7533|	0.9265|
|4|	8762.66|0.7551|	0.9260|
|8|	9184.62|0.7511|	0.9239|
|16|9431.46|0.7429|	0.9206|
|ADACOMM|8945.74|0.7555|0.9270|

可以看到在 navie local SGD （固定同步间隔）情况下，更新间隔越长训练的吞吐越高，但是模型的最终进度也会损失越大。 当使用 ADAPTIVE COMMUNICATION 策略后，训练在吞吐和精度间达到了一个更好的平衡。

## local SGD 快速开始
下文将以在单机8卡中训练 ResNet50 为例子简单介绍 Fleet 中 local SGD 的用法。 需要注意的是 单机八卡的通信都在同一节点内， 一般情况下参数同步并不会成为训练的瓶颈， 这里只是以其为例子，介绍Fleet 中 local SGD 参数的设置。

#### 添加依赖

```python
import os
import fleetx as X
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import time
import paddle.distributed.fleet as fleet

```

#### 定义分布式模式并初始化

通过`X.parse_train_configs()`接口，用户可以定义训练相关的参数，如：学习率、衰减率等。同时通过`fleet.init()`接口定义了分布式模型，下面代码中的`is_collective=True`表示采用集合通信的GPU分布式模式训练模型。
```python
configs = X.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)
```

#### 加载模型及数据

用户可以通过`X.applications`接口加载我们预先定义好的模型，如：Resnet50、VGG16、BERT等。并使用定制化的data_loader加载模型，同时可以定义训练中使用的batch_size等参数。
```python
model = X.applications.Resnet50()
loader = model.load_imagenet_from_file("/pathto/ImageNet/train.txt")
```

#### 定义local SGD 相关策略

用户首先需要定义paddle SGD 对象，并在SGD 对象中设置学习率参数。Fleet local SGD 中只有两个用户设置参数 `auto` 和 `k_step`，局部更新和参数同步都由框架自动完成：

* 在Naive local SGD 中： `auto = Flase`， 用户需要设置一个固定的常数 `k_step` 作为训练过程中的全局参数更新间隔。
* 在 自适应步长 local SGD中： `auto = True`， 用户需要设置`k_step` 作为第一次参数同步的间隔，之后的同步间隔将由上文中的公式动态确定，在学习率较大时，参数变化大，减小step，多进行通信从而保证快速收敛；在学习率较小时，参数变化小，增大step，减少通信次数，从而提升训练速度。 需要注意的是自适应步长策略中，系统会默认限制最大的同步间隔为 `16 steps`，当公式计算出的间隔大于16 时，按16 steps 进行参数同步。

```python
dist_strategy = fleet.DistributedStrategy()

dist_strategy.localsgd = True
dist_strategy.auto = True
dist_strategy.localsgd_configs = {
                    "k_steps": 1,
                }
optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)
```

#### 开始训练
这一部分和FleetX 中其他任务基本相同:

```python
place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

total_time = 0
for i, data in enumerate(data_loader()):
    start_time = time.time()
    cost_val = exe.run(paddle.static.default_main_program(),
                       feed=data,
                       fetch_list=[model.loss.name])
    end_time = time.time()
    total_time += (end_time - start_time)
    print(
        "worker_index: %d, step%d cost = %f, total time cost = %f, step per second: %f, speed: %f"
        % (fleet.worker_index(), i, cost_val[0], total_time,
           (i - 9) / total_time, 1 / (end_time - start_time))
```

### 运行训练脚本

一行启动单机多卡分布式训练：
```sh
fleetrun --gpus 0,1,2,3,4,5,6,7 resnet50_localsgd.py
```
