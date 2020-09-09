# 自动混合精度练加速分布式训练

## 简介
在使用数据并行分布式训练的同时, 我们还可以引入自动混合精度(Auto Mixed Precision) 来进一步提升训练的速度.

主流的神经网络模型通常使用单精度 `single-precision` `(FP32)` 数据格式来存储模型参数、进行训练和预测.
在上述环节中使用半精度 `half-precision` `(FP16)`来代替单精度. 可以带来以下好处:

1. 减少对GPU memory 的需求: 
    * 降低显存读写时的带宽压力 
    * GPU 显存不变情况下, 支持更大模型 / batch size
2. 加速GPU 数学运算速度 (需要GPU 支持[[1]](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensorop))
    * GPU上 FP16 吞吐是FP32 的 2 - 8 倍[[2]](https://arxiv.org/abs/1710.03740)

FleetX 支持自动混合精度计算, 并实现了 `FP32 参数副本及更新`,  `Dynamic loss scaling`,  `op黑白名单` 等功能来避免FP16 因动态范围较小而可能带来的模型最终精度损失. 
FleetX 并提供了简单易用的API 接口, 用户无须修改参数. 就可将自动混合精度应用到原有的分布式训练中进一步提升训练速度.

中下文将通过一个简单例子介绍如如何通过 FleetX将实现混合精度的分布式训练, 另外给出我们使用 FleetX 进行同步训练加速的实践.

## FleetX 效果
环境: 4 机 32卡 V100-32GB

| imagenet | 单卡 batch size | 速度 img/s | top1 |
|:---:|:---:|:---:|:---:|
|[VGG16-FP32](https://arxiv.org/abs/1708.03888)| 128 | TBA |  76.3% |
|[VGG16-AMP](https://arxiv.org/abs/1708.03888)| 128 | TBA |  76.3% |
|[VGG16-FP32](https://arxiv.org/abs/1708.03888)| 256 | OOM |  OOM|
|[VGG16-AMP](https://arxiv.org/abs/1708.03888)| 256 | TBA |  76.3% |

| imagenet | 单卡 batch size | 速度 img/s | top1 |
|:---:|:---:|:---:|:---:|
|[Resnet50-FP32](https://arxiv.org/abs/1708.03888)| 128 | 8410 |  TBA |
|[Resnet50-AMP](https://arxiv.org/abs/1708.03888)| 128 | TBA |  76.3% |
|[Resnet50-FP32](https://arxiv.org/abs/1708.03888)| 256 | OOM |  OOM |
|[Resnet50-AMP](https://arxiv.org/abs/1708.03888)| 256 | 29440 |  76.3% |

## AMP 快速开始
这里以在单机多卡上训练Resent50 为简单例子介绍FleetX 中 AMP的用法.
#### AMP 简述

##### FP32 参数副本及更新

<img src='./img/AMP_1.png' width = "952" height = "483" align="middle" description="xxxxxxxxxx" />

如上图所示, 在AMP 中, 模型参数 `weight` , 前向中间的结果`activation`, 反向的`gradient` 都以FP16 形式存储, 由此可以少显存所需空间和读写所需带宽. 
Paddle框架会为每一个`weight` 维护一个FP32副本, 用于参数更新. 上述点数据格式转换由 FleetX 框架自动完成, 用户无须操心. 

##### Loss scaling
<img src='./img/AMP_2.png' width = "973" height = "613" align="middle" description="xxxxxxxxxx" />

如上图所示, 实际情况中模型训练中的某些变量, 比如`grad` (特别是 `activation` 的 `grad`), 可能会因小于 FP16 所能表示的最小值而变成`0`; 

另一方面在FP16 的表示范围的中有很大的一部分(从最大值往左) 却没有被利用到.

对gradient 做一个整体的放大, 能够更充分的利用FP16 的表示范围. 

FleetX AMP 会在反向开始前对 loss 进行 up scaling, 并在执行任何 gradient-related 操作(e.g. gradient-clip, update) 之前对 gredient 进行  down scaling 恢复原来的大小. 

`scaling factor` 的设置是 Lossing scaling 的关键, FleetX AMP 提供 `Dynamic loss scaling` 和 `Constant loss scaling` 两种scaling 策略:

* Constant loss scaling: 设置 `use_dynamic_loss_scaling = False` 和 `init_loss_scaling (float)`
* Dynamic loss scaling: scaling 中面临的问题是当`scaling up 不足`时, 仍会有部分较小变量会被表示成 0而损失精度; 当`scaling up 过度`时, 变量超过FP16表示范围出现 nan or inf, 同样造成精度损失.  此策略采用自动 gradient 值检测的方式: 
    * 当连续`incr_every_n_steps(int)`个batch 中所有的gradient 都在FP16 的表示范围, 将scaling factor 增大`incr_ratio(float)`倍; 
    * 当有连续`decr_every_n_nan_or_inf(int)`个batch 中gradient 里出现 nan / inf时, scaling factor 缩小 `decr_ratio(float)`倍. 
    * 上述四个参数FleetX 提供的默认值可以满足绝大部分要求, 用户通常不需要修改.

如下图所示在 Dynamic loss scaling 中，框架在每一个 iteration 都会依据当前 gradients 是否出现 `nan` or `inf` 还有用户设置的 Dynamic loss scaling 参数来动态调整 loss scaling factor 的大小，将gradient 尽量保持在 FP16 的表示范围之内。

<img src='./img/AMP_3.png' width = "1930" height = "430" align="middle" description="xxxxxxxxxx" />

##### OP 黑白名单
模型中的某些`Operation (OP)` 可能对精度较为敏感, 为了确保AMP 中精度无损, 可以通过`OP 黑白名单`对具体OP 操作的精度做指定.

 * 白名单: OP 操作在FP16精度下进行, `input`: 如果不是FP16 会被首先cast 成FP16后再输入OP. `output`: FP16
 * 黑名单: OP 操作在FP32精度下进行, `input`: 如果不是FP32 会被首先cast 成FP32后再输入OP. `output`: FP32
 * 灰名单: 所有不在黑或白名单里的OP. 仅当OP 所有 inputs 都是 FP16精度时, 操作才在FP16精度下进行, 否着以FP 32进行. `input / output`: 和原始输入中的最高精度相同

FleetX 已经预设了一个能够覆盖绝大多数模型OPs的黑白名单, 通常情况下用户并不需要修改, 但是如果任务对精度有特殊要求, 或者希望新增自定义 OP, 用户可以通过 paddle.distributed.fleet.DistributedStrategy.amp_configs 中的 `custom_white_list` 和 `custom_black_list` 进行指定. 同是, 用户还可以通过`custom_black_varnames`, 来具体指定`Paddle program` 某一个 `var`必须使用FP32精度.

我们将在文末的 appendix中 进一步介绍 FleetX 的黑白名单设置及其影响。

#### 构建模型
首先我们要导入依赖和定义模型和 data loader, 这一步和FleetX 下其他任务基本一致.

```python
import os
import fleetx as X
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import time
import paddle.distributed.fleet as fleet

model = X.applications.Resnet50()
loader = model.load_imagenet_from_file("/pathto/ImageNet/train.txt")
```
#### 定义分布式及AMP 相关策略
如上文描述, 用户可以选择设置 `Loss scaling` 和 `OP黑白名单`等的参数. 

另外 FleetX 将AMP 实现为 meta optimizer, 用户需要指定其的 `inner-optimizer`.  FleetX AMP支持所有 paddle optimziers 和 FLeetX meta otpimizers 作为其 inner-optimizer. 

```python
configs = X.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)
dist_strategy = fleet.DistributedStrategy()

optimizer = fluid.optimizer.Momentum(learning_rate=configs.lr, momentum=configs.momentum)

dist_strategy = fleet.DistributedStrategy()
dist_strategy.amp = True

optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)

dist_strategy = paddle.distributed.fleet.DistributedStrategy()
dist_strategy.amp = True
dist_strategy.amp_configs = {
    "init_loss_scaling": 32768,
    "decr_every_n_nan_or_inf": 2,
    "incr_every_n_steps": 1000,
    "incr_ratio": 2.0,
    "use_dynamic_loss_scaling": True,
    "decr_ratio": 0.5,
    "custom_white_list": ['softmax'],
    "custom_black_list": ['tanh'],
}

optimizer = fluid.optimizer.Momentum(learning_rate=configs.lr, momentum=configs.momentum)
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
fleetrun --gpus 0,1,2,3,4,5,6,7 resnet50_amp.py
```

### FleetX 黑白名单设置
上文简要介绍了FleetX 中黑白名单的 API 接口， 下文将进一步介绍 FleetX 中黑白名单的实现和可能对训练造成影响。
目前 FleetX 中 AMP 的默认黑白名单如下， 其他未列出的 op 都属于灰名单：

```python
white_list = {
    'conv2d',
    'matmul',
    'mul',
}
black_list = {
    'exp',
    'square',
    'log',
    'mean',
    'sum',
    'cos_sim',
    'softmax',
    'softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
    'cross_entropy',
    'cross_entropy2',
}
```
#### 黑白名单设置
白名单中只有卷积和乘法运算，这样的设置能够满足大部分的 CV 场景的模型加速（Vgg、ResNet）， 因为卷积计算占据这些模型计算和内存访问开销的很大一部分， 其他 ops 的开销只占很小一部分。 对于 主要开销在 RNN 计算的 NLP 模型，目前的 AMP 实现提速并不是很明显。

黑名单中的 op 可以分为3 大类： 
    * 对精度非常敏感的 op： `softmax`，`cross_entropy` 等。
    * 输出相对于输入有更大动态范围的op（f(x) >> x）：`exp`，`square`, `log` 等。
    * reduce 类型的op： `mean`，`sum` 等。
所以，用户希望判断新的自定义op是否需要加入黑名单时，可以参考上述3个类型。

需要注意: 一些常用的 op 如 `BatchNorm`， `pooling`， `relu` 属于灰名单，这意味着这些 op 的数据类型决定于之前的 op 的类型； 另外并行分布式计算使用 AMP之后，gradient-allreduce 是在FP16 中进行的。

#### 自动化op 插入
在训练开始前，框架会根据黑白名单在前向和反向网络自动插入 cast op， 如：
* 前向中插入 FP32toFP16 cast， 将 FP32 的layer parameter 副本 cast 成 FP16， 进行 FP16 conv 计算。
* 反向中插入 FP16toFP32 cast， 将等到的 FP16 gradient cast 成 FP32， 然后更新 FP32 的parameter 副本。

cast op 虽然会带来额外的开销， 但是在诸如 Vgg、ResNet 等主要由重复的 conv layer 串行的而成 CV 模型中， 只需要cast input 和 每一层的param，并不需要cast 模型的中间结果，这样 cast 操作带来的开销较少, 容易倍半精度计算带来的加速覆盖；但是如果模型的串行 layers 序列中存在较多的黑名单 op（e.g. conv --> log --> conv --> square --> conv）， 这样模型的中间结果需要进行多次 FP32toFP16 和 FP16toFP32 cast， cast 开销将会急剧增大，从而抵消半精度带来的加速。

#### 可能不适用 AMP 加速的情况
* RNN 为主的 NLP 模型
* 模型组网中有较多黑名单 op 的模型
* 对数据精度敏感的任务（Adversarial Attacking in ML）

#### 图像 Input Layout 格式
CV 模型训练时了达到最佳速度，不同场景下推荐使用不同[图像 Layout](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)：

* FP32：`NCHW`
* 自动混合精度： `NHWC`

```python
# when build dataloader 
loader = model.load_imagenet_from_file("./ImageNet/train.txt",
                                        batch_size=args.batch_size,
                                        data_layout="NHWC")

# when build model  
if data_format == "NHWC":
    img_shape = [None, 224, 224, 3]
else:
    img_shape = [None, 3, 224, 224]
image = fluid.data( name="feed_image", shape=img_shape, dtype="float32", lod_level=0)
conv = fluid.layers.conv2d(input=input, data_format= "NHWC")
```


## 推荐阅读:
如果需要对自动混合精度做定制化修改,或更深入理解AMP中原理和实现推荐阅读:

* [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
* [MIXED PRECISION TRAINING: THEORY AND PRACTICE](https://on-demand.gputechconf.com/gtc/2018/presentation/s8923-training-neural-networks-with-mixed-precision-theory-and-practice.pdf)
* [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensorop)
