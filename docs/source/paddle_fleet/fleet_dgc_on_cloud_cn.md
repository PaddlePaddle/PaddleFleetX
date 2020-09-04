## DGC 优化低配网络的分布式GPU训练

## 简介 
大规模分布式训练需要较高的网络带宽以便进行梯度的聚合更新，这限制了多节点训练时的可扩展性同时也需要昂贵的高带宽设备。在低带宽云网络等环境下进行分布式训练会变得更加糟糕。 [Deep Gradient Compression](https://arxiv.org/abs/1712.01887) 发现：分布式SGD中有99.9%的梯度交换都是冗余的，可以使用深度梯度压缩选择重要梯度进行通信来减少通信量，降低对通信带宽的依赖。FleetX 实现了DGC的稀疏通信方式，可有效在低配网络下进行GPU分布式训练。FleetX 实现了 DGC 论文中的 `预热训练 (warming up training)`, `动量修正 (Momentum Correction)`, `局部梯度修剪 (local gradient clipping)`, `动量因子掩藏 (Momentum factor masking)` 等策略， 和 `正则化项修正 (Weight Decay Correction)` 避免稀疏梯度通信训练带来的最终模型精度损失。 

下面将介绍 DGC 稀疏通信方式的适用场景及、基本原理，FleetX 中 DGC 的效果和使用方法。

#### 适用场景
DGC稀疏通信在低带宽通信瓶颈时会有较大的性能提升，但**在单机多卡及RDMA网络通信并非瓶颈情况下**，并不会带来性能上的提升。同时由于AllGather的通信量会随卡数的增多而增大，所以DGC的多机训练规模也不宜过大。故DGC适用于低配网络，同时节点规模不宜过大，如>128张卡。在云网络或高带宽网络设备昂贵时，DGC可有效降低训练成本。

## FleetX 效果
* 模型：FasterRCNN
* 硬件： P40两机分布式，每台机器一卡，TCP网络测试。
* 取300-700步耗时/400step。
* 精度无损。

| 带宽 |训练耗时-Momentum （step /s）|训练耗时-DGCMomentum （step /s)| 比率 |
|:---:|:---:|:---:|:---:|
|100G|0.3725|0.375|0.993|
|10G|0.55|0.375|1.467|
|1G|0.55|0.375|6.533|

## DGC 原理

### 梯度稀疏
DGC的基本思路是通过只传送重要梯度，即只发送大于给定阈值的梯度来减少通信带宽的使用。为避免信息的丢失，DGC会将剩余梯度在局部累加起来，最终这些梯度会累加大到足以传输。
换个角度，从理论依据上来看，局部梯度累加等同于增大batch size，（DGC相当于每一个梯度有自己的batch size）。设定 $F(w)$ 为需要优化的loss函数，则有着N个训练节点的同步分布式SGD更新公式如下：

$$
F(w)=\\frac{1}{\|\\chi\|}\\sum\_{x\\in\\chi}f(x, w), \\qquad w\_{t+1}=w\_{t}-\\eta\\frac{1}{N b}\\sum\_{k=1}^{N}\\sum\_{x\\in\\mathcal{B}\_{k,t}}\\nabla f\\left(x, w\_{t}\\right) \\tag{1}
$$

其中$\chi$是训练集，$w$是网络权值，$f(x, w)$是每个样本$x \in \chi$的loss，$\eta$是学习率，N是训练节点个数，$\mathcal{B}\_{k, t}$代表第$k$个节点在第$t$个迭代时的minibatch，大小为b。
考虑权重的第i个值，在T次迭代后，可获得:

$$
w\_{t+T}^{(i)}=w\_{t}^{(i)}-\\eta T \\cdot \\frac{1}{N b T} \\sum\_{k=1}^{N}\\left(\\sum\_{\\tau=0}^{T-1} \\sum\_{x \\in \\mathcal{B}\_{k, t+\\tau}} \\nabla^{(i)} f\\left(x, w\_{t+\\tau}\\right)\\right)  \\tag{2}
$$

等式2表明局部梯度累加可以被认为batch size从$Nb$增大为$NbT$，其中T是$w^{(i)}$两次更新的稀疏通信间隔。

### 预热调参
对于正常的训练，使用DGC一般需进行预热训练，否则可能会有精度损失。如下图是ResNet50模型Imagenet数据集的训练结果，未进行预热训练的DGC最终损失了约0.3%的精度。
<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_guide/performance_improving/multinode_training_improving/images/dgc_resnet50_acc1.png" width="400"/>
</p>

预热训练调参可参照论文的设置。论文中使用了75%, 93.75%, 98.4375%, 99.6%, 99.9%稀疏度逐渐提升的策略。由于paddle稀疏梯度聚合通信使用了AllGather，通信量会随卡数增加而增长，所以在卡数较多时不推荐较低稀疏度的预热训练。如75%稀疏度时每张卡会选择25%的梯度进行通信，卡数为32时通信量是正常dense通信的32\*(1-0.75)=8倍，所以前几个epoch使用正常的dense通信为佳。可参照如下设置参数：

``` python
# 1. 以1252个step为一个epoch，前2个epochs使用正常dense通信，后3个epochs逐步提升稀疏度为99.9%
strategy.dgc_configs = {
    "rampup_begin_step": 1252*2,
    "rampup_step": 1252*3,
    "sparsity": [0.984375, 0.996, 0.999]
}
# 2. 前面4个epochs都使用dense通信，之后默认0.999稀疏度运行
strategy.dgc_configs = {
    "rampup_begin_step": 1252*4,
    "rampup_step": 1,
    "sparsity": [0.999]
}
```

对于Fine-tuning训练，现测试可无需预热训练，从第0个epoch直接使用DGC即可。

``` python
# 从第0步开始DGC稀疏通信
strategy.dgc_configs = {
    "rampup_begin_step": 0,
    "rampup_step": 1,
    "sparsity": [0.999]
}
```

### 局部梯度累加改进
正常情况，稀疏更新会严重影响收敛性。DGC中采用动量修正(Momentum Correction)和局部梯度裁减(Local Gradient Clipping)来解决这个问题。
#### 动量修正
有着N个节点分布式训练中vanilla momentum SGD公式，
$$
u\_{t}=m u\_{t-1}+\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right), \\quad w\_{t+1}=w\_{t}-\\eta u\_{t}  \\tag{3}
$$
其中$m$是动量因子，$N$是节点数，$\\nabla\_{k, t}=\\frac{1}{N b} \\sum\_{x \\in \\mathcal{B}\_{k, t}} \\nabla f\\left(x, w\_{t}\\right)$。
考虑第i个权重$w^{(i)}$，在T次迭代后，权重更新公式如下，
$$
w\_{t+T}^{(i)}=w\_{t}^{(i)}-\\eta\\left[\\cdots+\\left(\\sum\_{\\tau=0}^{T-2} m^{\\tau}\\right) \\nabla\_{k, t+1}^{(i)}+\\left(\\sum\_{\\tau=0}^{T-1} m^{\\tau}\\right) \\nabla\_{k, t}^{(i)}\\right]  \\tag{4}
$$
如果直接应用动量SGD到稀疏梯度更新中，则有公式，
$$
v\_{k, t}=v\_{k, t-1}+\\nabla\_{k, t}, \\quad u\_{t}=m u\_{t-1}+\\sum\_{k=1}^{N} \\operatorname{sparse}\\left(v\_{k, t}\\right), \\quad w\_{t+1}=w\_{t}-\\eta u\_{t} \\tag{5}
$$
其中$v\_k$是训练节点k上的局部梯度累加项，一旦$v\_k$大于某一阈值，则会在第二项中压缩梯度进行动量更新，并使用sparse()函数获得mask清空大于阈值的梯度。
$w^{(i)}$在T次稀疏更新后的权重为,
$$
w\_{t+T}^{(i)}=w\_{t}^{(i)}-\\eta\\left(\\cdots+\\nabla\_{k, t+1}^{(i)}+\\nabla\_{k, t}^{(i)}\\right) \\tag{6}
$$
相比传统动量SGD，方程6缺失了累积衰减因子$\sum\_{\tau=0}^{T-1} m^{\tau}$，会导致收敛精度的损失。如下图(a)，正常梯度更新从A点到B点，但是方程6则从A点到C点。当稀疏度很高时，会显著降低模型性能，所以需要在方程5基础上对梯度进行修正。
<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_guide/performance_improving/multinode_training_improving/images/dgc_without_momentum_correction.png" width="320"/>
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_guide/performance_improving/multinode_training_improving/images/dgc_with_momentum_correction.png" width="320"/>
</p>

若将方程3中速度项$u\_t$当作“梯度”，则方程3第二项可认为是在”梯度“$u\_t$上应用传统SGD，前面已经证明了局部梯度累加在传统SGD上是有效的。因此，可以使用方程3局部累加速度项$u\_t$而非累加真实的梯度$\nabla\_{k, t}$来修正方程5，
$$
u\_{k, t}=m u\_{k, t-1}+\\nabla\_{k, t}, \\quad v\_{k, t}=v\_{k, t-1}+u\_{k, t}, \\quad w\_{t+1}=w\_{t}-\\eta \\sum\_{k=1}^{N} \\operatorname{sparse}\\left(v\_{k, t}\\right)  \\tag{7}
$$

#### 局部梯度修剪
梯度修剪是防止梯度爆炸的常用方法。这方法由Pascanu等人在2013年提出，当梯度的l2-norms和大于给定阈值时，就对梯度rescale。正常梯度修剪在梯度聚合后使用，而DGC因为每个节点独立的进行局部梯度累加，所以DGC在使用$G\_t$累加前对其进行局部梯度修剪。阈值缩放为原来的$N^{-1/2}$
$$
thr\_{G^{k}}=N^{-1 / 2} \\cdot thr\_{G}  \\tag{8}
$$
### 克服迟滞效应
因为推迟了较小梯度更新权重的时间，所以会有权重陈旧性问题。稀疏度为99.9%时大部分参数需600到1000步更新一次。迟滞效应会减缓收敛并降低模型精度。DGC中采用动量因子掩藏和预热训练来解决这问题。

#### 动量因子掩藏
DGC中使用下面方程来掩藏动量因子减缓陈旧性问题。
$$
Mask \\leftarrow\\left|v\_{k, t}\\right|>t h r, \\quad v\_{k, t} \\leftarrow v\_{k, t} \\odot \\neg Mask, \\quad u\_{k, t} \\leftarrow u\_{k, t} \\odot \\neg Mask \\tag{9}
$$
此掩码可以停止延迟梯度产生的动量，防止陈旧梯度把权重引入错误的方向。

### 正则化(Weight Decay)项修正
Paddle框架以Weight Decay的形式实现正则化。以L2Decay为例，公式(3)中传统momentum添加weight decay后公式为
$$
G\_{t}=\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right)+\\lambda w\_{t}, \\quad  u\_{t}=m u\_{t-1}+G\_{t}, \\quad w\_{t+1}=w\_{t}-\\eta u\_{t} \\tag{10}
$$
其中$\lambda$为Weight Decay系数，$G\_{t}$为添加L2Decay项之后的聚合梯度。由于在公式7中进行了局部动量修正，所以按照相同思路在局部梯度上运用修正的Weight Decay项。如下公式在局部梯度上添加局部Weight Decay项即可。
$$
\\nabla\_{k, t}=\\nabla\_{k, t}+\\frac{\\lambda}{N} w\_{t} \\tag{11}
$$
在模型实际训练中，通常会设置weight decay的系数$\lambda=10^{-4}$，在卡数较多如4机32卡的情况下局部weight decay系数为$\frac{\lambda}{N}=\frac{10^{-4}}{32}=3.125\*10^{-6}$，在数值精度上偏低，测试训练时会损失一定精度。为此还需对局部weight decay项进行数值修正。如下公式，
$$
\\nabla\_{k, t}^{'}=N \\nabla\_{k, t}+\\lambda w\_{t}, \\quad
G\_{t}^{'}=\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}^{'}\\right)=N\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right)+N\\lambda w\_{t}, \\quad
G\_{t}=\\frac{G\_{t}^{'}}{N}=\\sum\_{k=1}^{N}\\left(\\nabla\_{k, t}\\right)+\\lambda w\_{t} \\tag{12}
$$
具体做法为对局部梯度乘以卡数求得$\nabla\_{k, t}^{'}$，此时$\lambda$项则无需除以卡数，聚合梯度求得$G\_{t}^{'}$再对聚合梯度除以卡数得到$G\_{t}$即可。

上述策略已经在框架中实现，用户无须设置。

## DGC 快速开始
下文以单机八卡上训练ResNet50 为例子简单介绍 FleetX 中 DGC 的使用。 因为 8张 GPU 的通信都在同一节点内， 一般情况下梯度通信并不会成为训练的瓶颈， 这里只是以其为例子，介绍FleetX 中 DGC 参数的设置。

**注意**：

* 使用DGC时需确保 fleet.DistributedStrategy.fuse_all_reduce_ops=False， 关闭fuse (现有fuse策略会造成DGC失效)。
* 硬件环境要求： DGC目前只支持GPU多卡及分布式collective训练，需要有相应的cuda、cuDNN、nccl环境。
* Paddle环境要求： DGC只支持GPU，所以需GPU版本的Paddle。

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

#### DGC 相关策略
这里假设：1252个step为一个epoch，前2个epochs使用正常dense通信，后3个epochs逐步提升稀疏度为99.9%

* `rampup_begin_step (int)`：DGC(含预热训练)开始的 step 
* `rampup_step (int)`：DGC中预热训练持续的 step. 如果sparsity 是 [0.75, 0.9375, 0.984375, 0.996, 0.999]，rampup_step 设成 100时， 在 0~19 steps 时 sparsity=0.75，在 20~39 steps 时 sparsity=0.9375， 以此类推。
* `sparsity (list[float])`：稀疏度 threshold, (1 - current sparsity) % 的gradient 将会被 allreduce。

```python
dist_strategy = fleet.DistributedStrategy()

dist_strategy.lars = True
dist_strategy.dgc_configs = {
    "rampup_begin_step": 1252*2,
    "rampup_step": 1252*3,
    "sparsity": [0.984375, 0.996, 0.999]
}

optimizer = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
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
fleetrun --gpus 0,1,2,3,4,5,6,7 resnet50_dgc.py
```
