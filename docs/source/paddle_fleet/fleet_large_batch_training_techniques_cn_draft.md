# 使用超大Batch进行训练

## 简介 + strategy列表

为了追求模型的性能不断提升，人们对更大规模的数据集、更深的网络层、更庞大的参数规模趋之若鹜。但是随之而来的就是给模型训练带来了巨大的压力，因此分布式技术及定制化AI芯片应运而生。但在分布式训练中，经常会遇到显存或者内存不足的情况，通常是以下几点原因导致的：

- 输入的数据过大，例如视频类训练数据。
- 深度模型的参数过多或过大，所需的存储空间超出了内存/显存的大小。
- AI芯片的内存有限。

为了能正常完成训练，我们通常只能使用较小的Batch Size以降低模型训练中的所需要的存储空间，这将导致很多模型无法通过提高训练时的Batch Size来提高模型的精度。为了解决这个问题，Fleet中提供了两种策略，使得模型可以使用超大Batch的方式完成训练：

- ** Forward Recomputation Backpropagation（FRB）**


扩大训练batch大小的策略：Forward Recomputation Backpropagation (FRB) 以及 Gradient Merge。下面我们将基于BERT模型的实用样例，分别对这两个策略进行讲解。

在开始之前，我们需要准备训练数据及词表

```sh
wget --no-check-certificate https://fleet.bj.bcebos.com/Bertdata/train_data.tar.gz
tar -xf train_data.tar.gz
wget --no-check-certificate https://fleet.bj.bcebos.com/Bertdata/vocab.txt
```
## Forward Recompute Backpropagation

首先，我们来介绍Fleet中通过 Forward Recompute Backpropagation 策略增大 BERT 模型在分布式训练中 batch size 的方法（假设脚本名称为bert_recompute.py）：

### 添加依赖

```python
import os
import time
import paddle
import fleetx as X
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
```

### 定义分布式模式并初始化

通过`X.parse_train_configs()`接口，用户可以定义训练相关的参数，如：学习率、衰减率等。同时通过`fleet.init()`接口定义了分布式模型，下面代码中的`is_collective=True`表示采用集合通信的GPU分布式模式训练模型。
```python
configs = X.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)
```

### 加载模型及数据

用户可以通过`X.applications`接口加载我们预先定义好的模型，如：Resnet50、VGG16、BERT等。并使用定制化的data_loader加载模型，同时可以定义训练中使用的batch_size等参数。下面的例子中，我们使用了recompute对Bert_large模型所支持的最大batch_size。

```python
model = X.applications.Bert_large()

data_loader = model.load_digital_dataset_from_file(
    data_dir='./train_data',
    vocab_path='./vocab.txt',
    max_seq_len=512,
    batch_size=53,
)
```

### 定义Recompute Strategy 及 Optimizer

接下来我们就可以定义分布式训练中所应用到的策略了。Forward Recomputation Backpropagation（FRB）的思想是将深度学习网络切分为k个部分（segments）。对每个segment而言：前向计算时，除了小部分必须存储在内存中的Variable外，其他中间结果都将被删除；在反向计算中，首先重新计算一遍前向算子，以获得中间结果，再运行反向算子。简而言之，FRB和普通的网络迭代相比，多计算了一遍前向算子。

我们把切分网络的变量叫做checkpoints。那么该如何选择这些checkpoints呢？我们知道深度学习网络通常是由一个个模块串联得到的，比如ResNet-50由16个block串联而成， Bert-Large由24个transformer串联而成，以两个子模块中间的变量作为切分点就是一个很好的选择。 对于非串联的网络（比如含有大量shortcut结构的网络），FRB也支持对其做切分， 只是可能多耗费一点内存（用于存储shortcut的Variable）。

下面的例子中，为了使用Recompute策略，我们将`dist_strategy.recompute`设置为True 并设置我们事先定义好的checkpoints。

接下来用户需要定义训练中更新模型所用到的优化器，并使用`fleet.distributed_optimizer`接口将优化器转换为分布式模式。

最后运行`optimizer.minimize(model.loss)` 将反向计算的算子插入训练网络，我们就可以开始训练了。
```python
dist_strategy = fleet.DistributedStrategy()
# 使用Recompute，并设置checkpoints
dist_strategy.recompute = True
dist_strategy.recompute_configs = {"checkpoints": model.checkpoints}

optimizer = fluid.optimizer.Adam(learning_rate=configs.lr)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)
```

### 开始训练

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
           (i - 9) / total_time, 1 / (end_time - start_time)))
```

### 运行训练脚本
完成脚本的编写后我们就可以使用以下命令训练分布式模型：
```sh
fleetrun --gpus 0,1,2,3,4,5,6,7 bert_recompute.py
```
### 效果测试

我们在BERT模型上对recompute的效果进行了测试，使用Recompute后batch size可以扩大近3倍。与混合精度一起使用时，batch_size可以进一步扩大。

- **Bert_large**: 

|Model|Baseline|Recompute| Recompute + mixed precision|
|-----|-----|-----|-----|
|batch size| 14 | 53 | 87 |
|speed|18.2 sents/s| 12.88 sents/s| 19.14 sents/s |


## Gradient Merge

下面，我们介绍如何使用 Gradient Merge 来扩大BERT模型分布式训练中的 batch size（假设脚本名称为bert_gradient_merge.py）：

与 Forward Recompute Backpropagation 相同，我们首先要添加依赖，定义分布式模式并加载模型及数据。

### 添加依赖

```python
import os
import time
import paddle
import fleetx as X
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
```

### 定义分布式模式并初始化
```python
configs = X.parse_train_configs()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)
```

### 加载模型及数据

```python
model = X.applications.Bert_large()

data_loader = model.load_digital_dataset_from_file(
    data_dir='./train_data',
    vocab_path='./vocab.txt',
    max_seq_len=512,
    batch_size=13,
)
```


### 定义Gradient Merge Strategy 及 Optimizer

Gradient Merge 扩大 batch size 的方法为：将大batch的输入切分成若干小batch，并对这些小batch分别进行 “前向+反向” 网络计算从而得到梯度。其间会有一部分显存/内存用于存放梯度，对每个小batch计算出的梯度进行叠加，在计算完所有小batch后用累加的梯度对模型进行更新。

通过GradientMerge 策略，用户只需要定义大batch被分割的粒度便可以实现大batch训练的目的。

在下面的例子中，我们定义了分割粒度为13，并分4步完成一个大batch的训练，从而达到了batch size为52的训练。
```python
dist_strategy = fleet.DistributedStrategy()
# 使用Gradient merge策略并设置相关参数
dist_strategy.gradient_merge = True
dist_strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}
optimizer = fluid.optimizer.Adam(learning_rate=configs.lr)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)
```

### 开始训练

Gradient Merge 的训练代码与 Recompute 策略相同：

```python
place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

total_time = 0
for i, data in enumerate(data_loader()):
    start_time = time.time()
    cost_val = exe.run(fluid.default_main_program(),
                       feed=data,
                       fetch_list=[model.loss.name])
    end_time = time.time()
    total_time += (end_time - start_time)
    print(
        "worker_index: %d, step%d cost = %f, total time cost = %f, step per second: %f, speed: %f"
        % (fleet.worker_index(), i, cost_val[0], total_time,
           (i - 9) / total_time, 1 / (end_time - start_time)))
```

### 运行训练脚本

```sh
fleetrun --gpus 0,1,2,3,4,5,6,7 bert_gradient_merge.py
```
