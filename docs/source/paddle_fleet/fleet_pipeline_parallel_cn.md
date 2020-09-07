# 使用流水线并行进行训练

## 简介

随着多种神经网络加速设备和专用神经网络计算芯片的出现，采用异构设备训练模型逐渐成为一种趋势。以CPU和GPU异构训练为例，CPU设备的并行计算能力较弱，但其具备数百GB到数TB的内存容量；与之不同，GPU设备具有强大的并行计算能力，但其显存容量仅为数十GB。同时，网络模型的不同层对计算能力和存储容量的需求差异显著。例如，神经网络的embedding层可以理解为查表操作，对计算能力的要求较低，但对存储容量的需求较大；与之相反，卷积类操作通常对存储容量的需求较低，但对计算能力的需求较高。因此，若能够根据异构设备和模型层间的不同特性，对模型的不同层使用不同的计算设备，可以优化模型训练过程。


## 原理

流水线并行分布式技术与数据并行不同，通过将模型切分到多个计算节点，并采用流水线执行的方式，实现模型的并行训练。以下图为例，模型被切分为三个部分，并分别放置到不同的计算设备（第1层放置到设备0，第2、3层被放置到设备1，第四层被放置到设备2）；设备间通过通信的方式来交换数据。

<img src='./img/pipeline-1.png' width = "683" height = "408" align="middle" description="xxxxxxxxxx" />

具体地讲，前向计算过程中，输入数据首先在设备0中通过第1层的计算得到中间结果，并将其传输到设备1，然后由设备1计算第2层和第3层，经过最后一层的计算后得到最终的前向计算结果；反向传播过程中，第四层使用前向计算结果得到相应的梯度数据，并由设备2传输到设备1，一次经过第3层和第二层，将结果传至设备0，经过第1层的计算完成所有的反向计算。最后，各个设备上的网络层会更新参数信息。

如下图，为流水线并行中的时序图。简单的流水线并行方式下，任一时刻只有单个计算设备处于激活状态，其它计算设备则处于空闲状态，因此设备利用率和计算效率较差。

<img src='./img/pipeline-2.png' width = "683" height = "408" align="middle" description="xxxxxxxxxx" />

为了优化流水线并行的性能，我们可以将mini-batch切分成若干更小粒度的micro-batch，提升流水线并行的并发度，达到提升设备利用率和计算效率的目的。如下图所示，一个mini-batch被切分为4个micro-batch；前向阶段，每个设备依次计算单个micro-batch的结果；这种减小mini-batch的方式减少了每个设备完成一次计算的时间，进而增加了设备间的并发度。

<img src='./img/pipeline-3.png' width = "683" height = "408" align="middle" description="xxxxxxxxxx" />

下面我们将通过例子为您讲解如何使用pipeline策略在两张GPU上训练Resnet50模型。


## 使用样例

### 导入依赖

```python
import os
import argparse
import time
import math 
import paddle.fluid as fluid

```

### 定义模型

在本例子中，我们为您实现了Paddle中Resnet模型的实现。其中`conv_bn_layer`、`shortcut`以及`bottleneck_block`函数根据模型的规律将一些计算层打包，以避免模型定义中重复使用相同的代码。

`build_network`函数中定义了最终的函数，对于不同深度的模型，我们都将其切分层数成相同的两份并通过`device_guard`接口的区分分别放置到两张GPU卡上。

对于CPU设备，在使用`device_guard`时只需要指定设备类型，即`device_guard("cpu")`；对于GPU设备，除了指定设备类型外，还需要指定设备的id，如`device_guard("gpu:0")`。

```python
def conv_bn_layer(input,
                  num_filters,
                  filter_size,
                  stride=1,
                  groups=1,
                  act=None):
    conv = fluid.layers.conv2d(input=input,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=stride,
                               padding=(filter_size-1)//2,
                               groups=groups,
                               act=None,
                               bias_attr=False)
    return fluid.layers.batch_norm(input=conv,
                                   act=act)
 
def shortcut(input,
             ch_out,
             stride,
             is_first):
    ch_in = input.shape[1]
    if ch_in != ch_out or stride != 1 or is_first == True:
        return conv_bn_layer(input,
                             ch_out,
                             1,
                             stride)
    else:
        return input
 
 
def bottleneck_block(input,
                     num_filters,
                     stride):
    conv0 = conv_bn_layer(input=input,
                          num_filters=num_filters,
                          filter_size=1,
                          act='relu')
    conv1 = conv_bn_layer(input=conv0,
                          num_filters=num_filters,
                          filter_size=3,
                          stride=stride,
                          act='relu')
    conv2 = conv_bn_layer(input=conv1,
                          num_filters=num_filters*4,
                          filter_size=1,
                          act=None)
 
    short = shortcut(input,
                     num_filters*4,
                     stride,
                     is_first=False)
 
    return fluid.layers.elementwise_add(x=short,
                                        y=conv2,
                                        act='relu')

def build_network(input,
                  layers=50,
                  class_dim=1000):
    supported_layers = [50, 101, 152]
    assert layers in supported_layers
    depth = None
    if layers == 50:
        depth = [3, 4, 6, 3]
    elif layers == 101:
        depth = [3, 4, 23, 3]
    elif layers == 152:
        depth = [3, 8, 36, 3]
    num_filters = [64, 128, 256, 512]
     
    # 指定层所在的设备
    with fluid.device_guard("gpu:0"):
        conv = conv_bn_layer(input=input,
                             num_filters=64,
                             filter_size=7,
                             stride=2,
                             act='relu')
        conv = fluid.layers.pool2d(input=conv,
                                   pool_size=3,
                                   pool_stride=2,
                                   pool_padding=1,
                                   pool_type='max')
        for block in range(len(depth)//2):
            for i in range(depth[block]):
                conv = bottleneck_block(
                            input=conv,
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1)
    # 指定网络层所在的设备
    with fluid.device_guard("gpu:1"):    
        for block in range(len(depth)//2, len(depth)):
            for i in range(depth[block]):
                conv = bottleneck_block(
                            input=conv,
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1)
 
        pool = fluid.layers.pool2d(input=conv,
                                   pool_size=7,
                                   pool_type='avg',
                                   global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(
                    input=pool,
                    size=class_dim,
                    param_attr=fluid.param_attr.ParamAttr(
                        initializer=fluid.initializer.Uniform(-stdv, stdv)))
    return out
```

### 定义数据集及梯度更新策略

定义完模型后，我们可以继续定义训练所需要的数据，以及训练中所用到的更新策略。

```python
# 定义模型剩余部分
with fluid.device_guard("gpu:0"):
    image_shape = [3, 224, 224]
    image = fluid.layers.data(name="image",
                              shape=image_shape,
                              dtype="float32")
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[image, label],
            capacity=64,
            use_double_buffer=True,
            iterable=False)

fc = build_network(image)

with fluid.device_guard("gpu:1"):
    out, prob = fluid.layers.softmax_with_cross_entropy(logits=fc,
                                                        label=label,
                                                        return_softmax=True)
    loss = fluid.layers.mean(out)
    acc_top1 = fluid.layers.accuracy(input=prob, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=prob, label=label, k=5)
 
opt = fluid.optimizer.Momentum(0.1, momentum=0.9)
opt = fluid.optimizer.PipelineOptimizer(
                            opt,
                            num_microbatches=args.microbatch_num)
opt.minimize(loss)

# 定义data loader；在该例子中，我们使用随机生成的数据。
def train_reader():
    for _ in range(2560):
        image = np.random.random([3, 224, 224]).astype('float32')
        label = np.random.random([1]).astype('uint64')
        yield image, label
place = fluid.CUDAPlace(0)
data_loader.set_sample_generator(train_reader,
                                 batch_size=args.microbatch_size)


```

### 开始训练

```python
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
t1 = time.time()
data_loader.start()
exe.train_from_dataset(fluid.default_main_program())
t2 = time.time()
print("Execution time: {}".format(t2 - t1))
```
