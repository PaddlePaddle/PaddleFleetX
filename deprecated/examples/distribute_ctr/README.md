# 从零开始的分布式CTR-DNN
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。

跟随这篇文档，将从零开始，一步步教您如何利用PaddlePaddle Fluid高效且易用的Api，搭建单机ctr-dnn深度学习模型，并利用高阶分布式Api-Fleet将其升级为可以在CPU集群中运行的`参数服务器`模式分布式深度学习模型。在学习本篇文档后，您可以入门Paddle组网，Paddle分布式模型的搭建，了解CPU多线程全异步模式的启用方法。
#
## 目录
* [运行环境检查](#运行环境检查)  
* [代码地址](#代码地址)  
* [数据准备](#数据准备)  
    * [数据来源](#数据来源)  
    * [数据预处理](#数据预处理)  
    * [一键下载训练及测试数据](#一键下载训练及测试数据)  
* [模型组网](#模型组网)  
    * [数据输入声明](#数据输入声明)  
    * [CTR-DNN模型组网](#ctr-dnn模型组网)
        * [Embedding层](#embedding层)
        * [FC层](#fc层)
        * [Loss及Auc计算](#loss及auc计算)
* [dataset数据读取](#dataset数据读取)
    * [引入dataset](#引入dataset)
    * [如何指定数据读取规则](#如何指定数据读取规则)
    * [快速调试Dataset](#快速调试dataset)
* [单机训练](#单机训练)
    * [单机训练流程](#单机训练流程)
    * [在dataset模式下获取训练中的变量](#在dataset模式下获取训练中的变量)
    * [运行单机训练](#运行单机训练)
* [分布式训练——异步模式(Async)](#分布式训练异步模式async)
    * [区别一：数据需要分配到各个节点上](#区别一数据需要分配到各个节点上)
    * [区别二：每个节点需要扮演不同的角色](#区别二每个节点需要扮演不同的角色)
        * [共有的环境变量](#共有的环境变量)
        * [Pserver特有的环境变量](#pserver特有的环境变量)
        * [Trainer特有的环境变量](#trainer特有的环境变量)
    * [区别三 需要指定分布式训练的策略](#区别三-需要指定分布式训练的策略)
    * [区别四 需要区分Pserver与Trainer的运行流程](#区别四-需要区分pserver与trainer的运行流程)
    * [同步与异步？你可能需要了解的知识](#同步与异步你可能需要了解的知识)
    * [运行：本地模拟分布式](#运行本地模拟分布式)
        * [方法一 运行loc_cluster.sh脚本](#方法一-运行loc_clustersh脚本)
        * [方法二 通过paddle.distributed.launch_ps运行模拟分布式](#方法二-通过paddledistributedlaunch_ps运行模拟分布式)
* [分布式训练——同步模式(Sync)](#分布式训练同步模式sync)
    * [同步模式数据读取](#同步模式数据读取)
    * [同步训练策略配置](#同步训练策略配置)
    * [使用pyreader进行多轮训练](#使用pyreader进行多轮训练)
    * [运行：本地模拟分布式](#运行本地模拟分布式-1)
        * [方法一 运行loc_cluster.sh脚本](#方法一-运行loc_clustersh脚本-1)
        * [方法二 通过paddle.distributed.launch_ps运行模拟分布式](#方法二-通过paddledistributedlaunch_ps运行模拟分布式-1)
* [分布式训练——模型保存及增量训练](#分布式训练模型保存及增量训练)
    * [单机训练中模型的保存](#单机训练中模型的保存)
    * [分布式训练中模型的保存](#分布式训练中模型的保存)
    * [分布式增量训练](#分布式增量训练)
    * [预测——离线单机预测](#预测离线单机预测)
        * [构建预测网络及加载模型参数](#构建预测网络及加载模型参数)
         * [测试数据的读取](#测试数据的读取)
         * [AUC的清零步骤](#auc的清零步骤)
         * [运行Infer](#运行infer)
* [benchmark](#benchmark)
#
## 运行环境检查

- 请确保您的运行环境基于Linux，示例代码支持`unbuntu`及`CentOS`
- 运行环境中python版本高于`2.7`
- 请确保您的paddle版本高于`1.6.1`，可以利用pip升级您的paddle版本
- 请确保您的本地模拟分布式运行环境中没有设置`http/https`代理，可以在终端键入`env`查看环境变量

## 代码地址

- 示例代码位于：https://github.com/MrChengmo/Fleet/tree/re_ctr/examples/distribute_ctr
  
  在工作环境安装git后，在工作目录克隆Fleet代码仓库，示例代码位于`Fleet/example/distribute_ctr`
- 示例代码结构为：

   ```text
      .
      ├── get_data.sh               # 数据下载脚本
      ├── loc_cluster.sh            # 本地模拟分布式一键启动脚本
      ├── argument.py               # 超参数及环境变量配置
      ├── network.py                # CTR网络结构
      ├── local_train.py            # 单机训练示例代码
      ├── async_train.py            # 异步模式分布式训练示例代码
      ├── sync_train.py             # 同步模式分布式训练示例代码
      ├── infer.py                  # 模型测试示例代码
      ├── dataset_generator.sh      # dataset数据读取示例代码
      ├── py_reader_generator.py    # pyreader数据读取示例代码
      ├── README.md                 # 使用说明
   ```
#
## 数据准备
### 数据来源
训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
每一行数据格式如下所示：
```bash
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。

### 数据预处理
数据预处理共包括两步：
- 将原始训练集按9:1划分为训练集和验证集
- 数值特征（连续特征）需进行归一化处理，但需要注意的是，对每一个特征```<integer feature i>```，归一化时用到的最大值并不是用全局最大值，而是取排序后95%位置处的特征值作为最大值，同时保留极值。

### 一键下载训练及测试数据
```bash
sh get_data.sh
```
执行该脚本，会从国内源的服务器上下载Criteo数据集，并解压到指定文件夹。全量训练数据放置于`./train_data_full/`，全量测试数据放置于`./test_data_full/`，用于快速验证的训练数据与测试数据放置于`./train_data/`与`./test_data/`。

执行该脚本的理想输出为：
```bash
> sh get_data.sh
--2019-11-26 06:31:33--  https://fleet.bj.bcebos.com/ctr_data.tar.gz
Resolving fleet.bj.bcebos.com... 10.180.112.31
Connecting to fleet.bj.bcebos.com|10.180.112.31|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 4041125592 (3.8G) [application/x-gzip]
Saving to: “ctr_data.tar.gz”

100%[==================================================================================================================>] 4,041,125,592  120M/s   in 32s

2019-11-26 06:32:05 (120 MB/s) - “ctr_data.tar.gz” saved [4041125592/4041125592]

raw_data/
raw_data/part-55
raw_data/part-113
...
test_data/part-227
test_data/part-222
Complete data download.
Full Train data stored in ./train_data_full
Full Test data stored in ./test_data_full
Rapid Verification train data stored in ./train_data
Rapid Verification test data stored in ./test_data
```
至此，我们已完成数据准备的全部工作。

#
## 模型组网

### 数据输入声明
正如数据准备章节所介绍，Criteo数据集中，分为连续数据与离散（稀疏）数据，所以整体而言，CTR-DNN模型的数据输入层包括三个，分别是：`dense_input`用于输入连续数据，维度由超参数`dense_feature_dim`指定，数据类型是归一化后的浮点型数据。`sparse_input_ids`用于记录离散数据，在Criteo数据集中，共有26个slot，所以我们创建了名为`C1~C26`的26个稀疏参数输入，并设置`lod_level=1`，代表其为变长数据，数据类型为整数；最后是每条样本的`label`，代表了是否被点击，数据类型是整数，0代表负样例，1代表正样例。

在Paddle中数据输入的声明使用`paddle.fluid.layers.data()`，会创建指定类型的占位符，数据IO会依据此定义进行数据的输入。
```python
dense_input = fluid.layers.data(name="dense_input",
                                shape=[params.dense_feature_dim],
                                dtype="float32")

sparse_input_ids = [
   fluid.layers.data(name="C" + str(i),
                     shape=[1],
                     lod_level=1,
                     dtype="int64") for i in range(1, 27)
]

label = fluid.layers.data(name="label", shape=[1], dtype="int64")
inputs = [dense_input] + sparse_input_ids + [label]
```

### CTR-DNN模型组网

CTR-DNN模型的组网是比较简单的，本质是一个二分类任务，代码参考`network.py`。模型主要组成是一个`Embedding`层，三个`FC`层，以及相应的分类任务的loss计算和auc计算。

#### Embedding层
首先介绍Embedding层的搭建方式：`Embedding`层的输入是`sparse_input`，shape由超参的`sparse_feature_dim`和`embedding_size`定义。需要特别解释的是`is_sparse`参数，当我们指定`is_sprase=True`后，计算图会将该参数视为稀疏参数，反向更新以及分布式通信时，都以稀疏的方式进行，会极大的提升运行效率，同时保证效果一致。

各个稀疏的输入通过Embedding层后，将其合并起来，置于一个list内，以方便进行concat的操作。

```python
def embedding_layer(input):
   return fluid.layers.embedding(
            input=input,
            is_sparse=params.is_sparse,
            size=[params.sparse_feature_dim, 
                  params.embedding_size],
            param_attr=fluid.ParamAttr(
            name="SparseFeatFactors",
            initializer=fluid.initializer.Uniform()),
   )

sparse_embed_seq = list(map(embedding_layer, inputs[1:-1])) # [C1~C26]
```

#### FC层
将离散数据通过embedding查表得到的值，与连续数据的输入进行`concat`操作，合为一个整体输入，作为全链接层的原始输入。我们共设计了3层FC，每层FC的输出维度都为400，每层FC都后接一个`relu`激活函数，每层FC的初始化方式为符合正态分布的随机初始化，标准差与上一层的输出维度的平方根成反比。
```python
concated = fluid.layers.concat(sparse_embed_seq + inputs[0:1], axis=1)
        
fc1 = fluid.layers.fc(
   input=concated,
   size=400,
   act="relu",
   param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
         scale=1 / math.sqrt(concated.shape[1]))),
)
fc2 = fluid.layers.fc(
   input=fc1,
   size=400,
   act="relu",
   param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
         scale=1 / math.sqrt(fc1.shape[1]))),
)
fc3 = fluid.layers.fc(
   input=fc2,
   size=400,
   act="relu",
   param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
         scale=1 / math.sqrt(fc2.shape[1]))),
)
```
#### Loss及Auc计算
- 预测的结果通过一个输出shape为2的FC层给出，该FC层的激活函数时softmax，会给出每条样本分属于正负样本的概率。
- 每条样本的损失由交叉熵给出，交叉熵的输入维度为[batch_size,2]，数据类型为float，label的输入维度为[batch_size,1]，数据类型为int。
- 该batch的损失`avg_cost`是各条样本的损失之和
- 我们同时还会计算预测的auc，auc的结果由`fluid.layers.auc()`给出，该层的返回值有三个，分别是全局auc: `auc_var`，当前batch的auc: `batch_auc_var`，以及auc_states: `auc_states`，auc_states包含了`batch_stat_pos, batch_stat_neg, stat_pos, stat_neg`信息。`batch_auc`我们取近20个batch的平均，由参数`slide_steps=20`指定，roc曲线的离散化的临界数值设置为4096，由`num_thresholds=2**12`指定。
```
predict = fluid.layers.fc(
            input=fc3,
            size=2,
            act="softmax",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(fc3.shape[1]))),
        )

cost = fluid.layers.cross_entropy(input=predict, label=inputs[-1])
avg_cost = fluid.layers.reduce_sum(cost)
accuracy = fluid.layers.accuracy(input=predict, label=inputs[-1])
auc_var, batch_auc_var, auc_states = fluid.layers.auc(
                                          input=predict,
                                          label=inputs[-1],
                                          num_thresholds=2**12,
                                          slide_steps=20)
```

完成上述组网后，我们最终可以通过训练拿到`avg_cost`与`auc`两个重要指标。

#
## dataset数据读取
为了能高速运行CTR模型的训练，我们使用`dataset`API进行高性能的IO，dataset是为多线程及全异步方式量身打造的数据读取方式，每个数据读取线程会与一个训练线程耦合，形成了多生产者-多消费者的模式，会极大的加速我们的模型训练。

如何在我们的训练中引入dataset读取方式呢？无需变更数据格式，只需在我们的训练代码中加入以下内容，便可达到媲美二进制读取的高效率，以下是一个比较完整的流程：

### 引入dataset

1. 通过工厂类`fluid.DatasetFactory()`创建一个dataset对象。
2. 将我们定义好的数据输入格式传给dataset，通过`dataset.set_use_var(inputs)`实现。
3. 指定我们的数据读取方式，由`dataset_generator.py`实现数据读取的规则，后面将会介绍读取规则的实现。
4. 指定数据读取的batch_size。
5. 指定数据读取的线程数，该线程数和训练线程应保持一致，两者为耦合的关系。
6. 指定dataset读取的训练文件的列表。


```python
def get_dataset(inputs, params)
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python dataset_generator.py")
    dataset.set_batch_size(params.batch_size)
    dataset.set_thread(int(params.cpu_num))
    file_list = [
        str(params.train_files_path) + "/%s" % x
        for x in os.listdir(params.train_files_path)
    ]
    dataset.set_filelist(file_list)
    logger.info("file list: {}".format(file_list))
    return dataset
```

### 如何指定数据读取规则

在上文我们提到了由`dataset_generator.py`实现具体的数据读取规则，那么，怎样为dataset创建数据读取的规则呢？
以下是`dataset_generator.py`的全部代码，具体流程如下：
1. 首先我们需要引入dataset的库，位于`paddle.fluid.incubate.data_generator`。
2. 声明一些在数据读取中会用到的变量，如示例代码中的`cont_min_`、`categorical_range_`等。
3. 创建一个子类，继承dataset的基类，基类有多种选择，如果是多种数据类型混合，并且需要转化为数值进行预处理的，建议使用`MultiSlotDataGenerator`；若已经完成了预处理并保存为数据文件，可以直接以`string`的方式进行读取，使用`MultiSlotStringDataGenerator`，能够进一步加速。在示例代码，我们继承并实现了名为`CriteoDataset`的dataset子类，使用`MultiSlotDataGenerator`方法。
4. 继承并实现基类中的`generate_sample`函数，逐行读取数据。该函数应返回一个可以迭代的reader方法(带有yield的函数不再是一个普通的函数，而是一个生成器generator，成为了可以迭代的对象，等价于一个数组、链表、文件、字符串etc.)
5. 在这个可以迭代的函数中，如示例代码中的`def reader()`，我们定义数据读取的逻辑。例如对以行为单位的数据进行截取，转换及预处理。
6. 最后，我们需要将数据整理为特定的格式，才能够被dataset正确读取，并灌入的训练的网络中。简单来说，数据的输出顺序与我们在网络中创建的`inputs`必须是严格一一对应的，并转换为类似字典的形式。在示例代码中，我们使用`zip`的方法将参数名与数值构成的元组组成了一个list，并将其yield输出。如果展开来看，我们输出的数据形如`[('dense_feature',[value]),('C1',[value]),('C2',[value]),...,('C26',[value]),('label',[value])]`


```python
import paddle.fluid.incubate.data_generator as dg

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)

class CriteoDataset(dg.MultiSlotDataGenerator):
   
    def generate_sample(self, line):
        
        def reader():
            features = line.rstrip('\n').split('\t')
            dense_feature = []
            sparse_feature = []
            for idx in continuous_range_:
                if features[idx] == "":
                    dense_feature.append(0.0)
                else:
                    dense_feature.append(
                        (float(features[idx]) - cont_min_[idx - 1]) /
                        cont_diff_[idx - 1])
            for idx in categorical_range_:
                sparse_feature.append(
                    [hash(str(idx) + features[idx]) % hash_dim_])
            label = [int(features[0])]
            process_line = dense_feature, sparse_feature, label
            feature_name = ["dense_feature"]
            for idx in categorical_range_:
                feature_name.append("C" + str(idx - 13))
            feature_name.append("label")

            yield zip(feature_name, [dense_feature] + sparse_feature + [label])

        return reader

d = CriteoDataset()
d.run_from_stdin()
```
### 快速调试Dataset
我们可以脱离组网架构，单独验证Dataset的输出是否符合我们预期。使用命令
`cat 数据文件 | python dataset读取python文件`进行dataset代码的调试：
```bash
cat train_data/part-0 | python dataset_generator.py
```
输出的数据格式如下：
` dense_input:size ; dense_input:value ; sparse_input:size ; sparse_input:value ; ... ; sparse_input:size ; sparse_input:value ; label:size ; label:value `

理想的输出为(截取了一个片段)：
```bash
...
13 0.05 0.00663349917081 0.05 0.0 0.02159375 0.008 0.15 0.04 0.362 0.1 0.2 0.0 0.04 1 715353 1 817085 1 851010 1 833725 1 286835 1 948614 1 881652 1 507110 1 27346 1 646986 1 643076 1 200960 1 18464 1 202774 1 532679 1 729573 1 342789 1 562805 1 880474 1 984402 1 666449 1 26235 1 700326 1 452909 1 884722 1 787527 1 0
...
```

>使用Dataset的一些注意事项
> - Dataset的基本原理：将数据print到缓存，再由C++端的代码实现读取，因此，我们不能在dataset的读取代码中，加入与数据读取无关的print信息，会导致C++端拿到错误的数据信息。
> - dataset目前只支持在`unbuntu`及`CentOS`等标准Linux环境下使用，在`Windows`及`Mac`下使用时，会产生预料之外的错误，请知悉。

#
## 单机训练

当我们完成`数据准备`、`模型组网`及`数据读取`上述三个步骤后，就可以开始进行单机的训练，单机训练代码见`local_train.py`文件。

### 单机训练流程

```python
def train(params):
    # 引入模型的组网
    ctr_model = CTR()
    inputs = ctr_model.input_data(params)
    avg_cost, auc_var, batch_auc_var = ctr_model.net(inputs,params)
    
    # 选择反向更新优化策略
    optimizer = fluid.optimizer.Adam(params.learning_rate)
    optimizer.minimize(avg_cost)

    # 创建训练的执行器
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    
    # 引入数据读取
    dataset = get_dataset(inputs,params)

    # 开始训练
    for epoch in range(params.epochs):
        start_time = time.time()
        # 使用train_from_dataset实现多线程并发训练
        exe.train_from_dataset(program=fluid.default_main_porgram(),
                               dataset=dataset, fetch_list=[auc_var],
                               fetch_info=["Epoch {} auc ".format(epoch)],
                               print_period=10, debug=False)
        end_time = time.time()
        logger.info("epoch %d finished, use time=%d\n" % ((epoch), end_time - start_time))

        if params.test:
            model_path = (str(params.model_path) + "/"+"epoch_" + str(epoch))
            fluid.io.save_persistables(executor=exe, dirname=model_path)

    logger.info("Train Success!")
    return train_result
```
通过以上简洁的代码，即可以在单机上实现CTR模型的多线程并发训练。

### 在dataset模式下获取训练中的变量
dataset在提高了模型训练速度的同时，也会有其他的不便，比如不能像传统方式一样，在训练中，通过形如
```
auc = exe.run(fetch_list=[auc])
```
的方式实时获取每个mini_batch的参数变化。这会对我们的训练过程监控会造成一些不便。

> 如何获取dataset模式训练中的参数变化？
> 
> 在paddle dataset模式中，由于dataset设计初衷是保证高速，运行于程序底层，与paddlepaddle传统的`feed={dict}`方法不一致，不支持直接通过`train_from_dataset`的函数返回值，得到当前训练中`fetch_list`的值。
> 
> 但我们可以通过`paddle/release/1.6`中新增的`fetch_handler`方法创建一个新的线程，监听训练过程，不影响训练的效率。该方法需要继承`fluid.executor.FetchHandler`类中的`handler`方法实现一个监听函数。`var_dict`是一个dict，由我们自行指定哪些变量的值需要被监控。在`exe.train_from_dataset`方法中，指定`fetch_handler`为我们实现的监听函数。可以配置2个超参：
> - 第一个是`var_dict`，添加我们想要获取的变量的名称，示例中，我们指定为`{"auc": auc_var}`
> - 第二个是监听函数的更新频率，单位是s，示例中我们设置为5s更新一次。

改动后的训练代码如下
```python
for epoch in range(num_epochs):
      start_time = time.time()
      class fetch_vars(fluid.executor.FetchHandler):
          def handler(self, fetch_target_vars):
              auc_value = fetch_target_vars["auc"]
              logger.info(
                  "epoch -> {}, auc -> {}, at: {}".format(epoch, auc_value, time.ctime()))
      # 开始训练
      var_dict = {"auc": auc_var}
      exe.train_from_dataset(program=fleet.main_program,
                            dataset=dataset,
                            fetch_handler=fetch_vars(var_dict, 5))
      end_time = time.time()
```
如此，便可以在`def handler()`函数中实现训练过程的实时监控，但该监控值的打印频率，不是以`mini_batch`为单位，而是以时间`s`为单位，请知悉。`Fetch hadnler`所在的线程与训练线程并发，但不完全解耦。具体表现在：fetch线程在每个epoch开始时进行计时，epoch结束重置计时。训练线程结束时，会检测fetch线程是否仍在工作，会持续等待fetch线程运行结束。

### 运行单机训练
为了快速验证效果，我们可以用小样本数据快速运行起来，只取前两个part的数据进行训练。在代码目录下，通过键入以下命令启动单机训练。
```bash
python -u local_train.py --test=True &> train.log &
```
训练过程的日志保存在`./train.log`文件中。使用默认配置运行的理想输出为：
```bash
2019-11-26 07:11:34,977 - INFO - file list: ['train_data/part-1', 'train_data/part-0']
2019-11-26 07:11:34,978 - INFO - Training Begin
Epoch 0 auc     auc_0.tmp_0             lod: {}
        dim: 1
        layout: NCHW
        dtype: double
        data: [0.626496]

Epoch 0 auc     auc_0.tmp_0             lod: {}
        dim: 1
        layout: NCHW
        dtype: double
        data: [0.667014]

2019-11-26 07:12:27,155 - INFO - epoch 0 finished, use time=52

2019-11-26 07:12:27,549 - INFO - Train Success!
```

#
## 分布式训练——异步模式(Async)
PaddlePaddle在release/1.5.0之后新增了高级分布式API-`Fleet`，只需数行代码便可将单机模型转换为分布式模型。异步模式分布式训练代码见`async_train.py`，我们通过与单机训练的代码对比，来说明基于fleet将单机训练转换为分布式训练需要哪些步骤。

### 区别一：数据需要分配到各个节点上
单机训练中，我们没有对数据做过多的区分。但在分布式训练中，我们要确保每个节点都能拿到数据，并且希望每个节点的数据同时满足：1、各个节点数据无重复。2、各个节点数据均匀。Fleet提供了`split_files()`的接口，输入值是一个稳定的目录List，随后该函数会根据节点自身的编号拿到相应的数据文件列表。示例代码中，我们假设您在集群进行分布式训练，且设置`params.cloud=True`时，已经将文件分到了各个节点上。所以仅需要进行本地模拟分布式时，使用该接口，给各个进程分配不同的数据文件。

```python
file_list = [
        str(params.train_files_path) + "/%s" % x
        for x in os.listdir(params.train_files_path)
]
# 请确保每一个训练节点都持有不同的训练文件
# 当我们用本地多进程模拟分布式时，每个进程需要拿到不同的文件
# 使用 fleet.split_files 可以便捷的以文件为单位分配训练样本
if not param.cloud
    file_list = fleet.split_files(file_list)
dataset.set_filelist(file_list)
```

### 区别二：每个节点需要扮演不同的角色
单机训练流程中，程序完成了从`数据读取->前向loss计算->反向梯度计算->参数更新`的完整流程，但在分布式训练中，单节点不一定需要完成全部步骤，比如在`同步(Sync)`及`异步(Async)`模式下，`Trainer`节点完成`数据读取->前向loss计算->反向梯度计算`的步骤，而`Pserver`节点完成`参数更新`的步骤，两者分工协作，解决了单机不能训练大数据的问题。

因此，在分布式训练中，我们需要指定每个节点扮演的角色。使用Fleet下提供的`PaddleCloudRoleMaker()`接口可以很便捷的获取当前节点所扮演的角色。

```python
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet

# 根据环境变量确定当前机器/进程在分布式训练中扮演的角色
# 使用 fleet api的 init()方法初始化这个节点
role = role_maker.PaddleCloudRoleMaker()
fleet.init(role) #必不可少的步骤，初始化节点！
```
> PaddleCloudRoleMaker()是怎样判断当前节点所扮演的角色的？
> 
> Paddle参数服务器模式中，使用各个节点机器的环境变量来确定当前节点的角色。为了能准确无误的分配角色，在每个节点上，我们都需要指定如下环境变量：
> #### 共有的环境变量
> - export PADDLE_TRAINERS_NUM=2 # 训练节点数
> - export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:36011,127.0.0.1:36012" # 各个pserver的ip:port 组合构成的字符串
> 
> #### Pserver特有的环境变量
> - export TRAINING_ROLE=PSERVER # 当前节点的角色是PSERVER
> - export PADDLE_PORT=36011 # 当前PSERVER的通信端口
> - export POD_IP=127.0.0.1 # 当前PSERVER的ip地址
> #### Trainer特有的环境变量
> - export TRAINING_ROLE=TRAINER # 当前节点的角色是TRAINER
> - export PADDLE_TRAINER_ID=0 # 当前Trainer节点的编号,范围为[0，PADDLE_TRAINERS_NUM)
> 
> 完成上述环境变量指定后，`PaddleCloudRoleMaker()`便可以正常的运行，决定当前节点的角色。

### 区别三 需要指定分布式训练的策略

Paddle的`参数服务器`模式分布式训练有很多种类型，根据通信策略可以分为：`同步Sync`、`半异步Half-Async`、`异步Async`、`GEO-SGD`等。所以需要配置分布式的运行策略，并将该策略传入`Optimizer`，构建不同的运行`Program`：

```python
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

# 进一步指定分布式的运行模式，通过 DistributeTranspilerConfig进行配置
# 如下，设置分布式运行模式为异步(async)，同时设置参数需要切分，以分配到不同的节点
strategy = DistributeTranspilerConfig()
strategy.sync_mode = False
strategy.runtime_split_send_recv = True

ctr_model = CTR()
inputs = ctr_model.input_data(params)
avg_cost, auc_var, batch_auc_var = ctr_model.net(inputs,params)
optimizer = fluid.optimizer.Adam(params.learning_rate)
# 配置分布式的optimizer，传入我们指定的strategy，构建program
optimizer = fleet.distributed_optimizer(optimizer,strategy)
optimizer.minimize(avg_cost)
```
`sync_mode`表示是同步训练，还是异步训练。同步训练会给网络中增加`barrier op`来保证各个节点之间的训练速度是一致的，异步训练则没有。于此同时，相较于单机网络，分布式训练中trainer节点的网络会增加`send op`来发送参数梯度信息给pserver。在Fleet API中，为了提高代码的可读性以及简洁性，我们将分布式训练中参数通信的工作都封装到了`padddle.fluid.communicator.Communicator`中，`send op`仅在分布式组网阶段获取通信相关的信息，然后调用`Communicator`中的参数发送单元来实现真正的参数收发。`runtime_split_send_recv`就是用`Communicator`来完成分布式通信的过渡阶段中的一个配置参数，如果设置为True，则代码启用`Communicator`，反之则不启用，依旧通过`send op`来发送参数梯度。此处推荐启用，该参数在将来可能会被废弃。

### 区别四 需要区分Pserver与Trainer的运行流程
Fleet隐式的完成了Pserver与Trainer的Program切分逻辑，我们可以使用`fleet.main_program`与`fleet.startup_program`，替代`fluid.default_main_program()`与`fluid.default_startup_program()`，拿到当前节点的训练program与初始化program。如何让Pserver和Trainer运行起来呢？其逻辑略有区别，但也十分容易掌握：
> 启动Pserver

启动参数服务器端，如果需要从某个模型热启，在训练开始之前加载某次训练得到的参数，则只需将初始化模型路径传入`init_server()`函数即可
```python
# 根据节点角色，分别运行不同的逻辑
if fleet.is_server():
    # 初始化参数服务器节点
    fleet.init_server()
    # 运行参数服务器节点
    fleet.run_server()
```
> 启动Trainer
启动训练节点，训练节点首先调用`init_worker()`来完成节点初始化，然后执行`fleet.startup_program`，从服务器端同步参数的初始化值。接着，和本地训练完全一致，通过执行`fleet.main_program`来完成整个训练过程，并保存模型。最后调用`fleet.stop_worker()`关闭训练节点。
```python
elif fleet.is_worker():
    # 必不可少的步骤，初始化工作节点！
    fleet.init_worker()
    exe = fluid.Executor(fluid.CPUPlace())

    # 初始化含有分布式流程的fleet.startup_program
    exe.run(fleet.startup_program))
    
    # 引入数据读取dataset
    dataset = get_dataset(inputs,params)

    for epoch in range(params.epochs):
        start_time = time.time()
        # 训练节点运行的是经过分布式配置的fleet.mian_program
        exe.train_from_dataset(program=fleet.main_program,
                            dataset=dataset, fetch_list=[auc_var],
                            fetch_info=["Epoch {} auc ".format(epoch)],
                            print_period=10, debug=False)
        end_time = time.time()
        logger.info("epoch %d finished, use time=%d\n" % ((epoch), end_time - start_time))

        # 默认使用0号节点保存模型
        if params.test and fleet.is_first_worker():
            model_path = (str(params.model_path) + "/"+"epoch_" + str(epoch))
            fluid.io.save_persistables(executor=exe, dirname=model_path)
    
    # 训练结束，调用stop_worker()通知pserver
    fleet.stop_worker() 
    logger.info("Distribute Train Success!")
    return train_result
```

了解了以上区别，便可以非常顺利的将单机模型升级为分布式训练模型。

>### 同步与异步？你可能需要了解的知识
>在`区别三 需要指定分布式运行策略`中，我们简要的提及了目前Paddle参数服务器模式支持的分布式运行策略：`同步Sync`、`半异步Half-Async`、`异步Async`与`GEO-SGD`。这些名词您可能有些陌生，具体介绍可以参考文档[PaddlePaddle Fluid CPU分布式训练(Transplier)使用指南](https://github.com/PaddlePaddle/Fleet/blob/develop/markdown_doc/transpiler/transpiler_cpu.md)


### 运行：本地模拟分布式
如果暂时没有集群环境，或者想要快速调试代码，可以通过本地多进程模拟分布式来运行分布式训练的代码。
有两种方法可以进行本地多进程模拟分布式。
#### 方法 运行`loc_cluster.sh`脚本
示例代码中，给出了本地模拟分布式的一键启动脚本`loc_cluster.sh async`，在代码目录，通过命令
```bash
# 根据自己的运行环境，选择sh或bash
# 启动命令选择异步模式async
sh local_cluster.sh async
```
便可以开启分布式模拟训练，默认启用2x2的训练模式。Trainer与Pserver的运行日志，存放于`./log/`文件夹，保存的模型位于`./model/`，使用默认配置运行后，理想输出为：
> pserver.0.log
```bash
get_pserver_program() is deprecated, call get_pserver_programs() to get pserver main and startup in a single call.
I1126 07:37:49.952580 15056 grpc_server.cc:477] Server listening on 127.0.0.1:36011 successful, selected port: 36011
```

> trainer.0.log
```bash
I1126 07:37:52.812678 14715 communicator_py.cc:43] using communicator
I1126 07:37:52.814765 14715 communicator.cc:77] communicator_independent_recv_thread: 1
I1126 07:37:52.814792 14715 communicator.cc:79] communicator_send_queue_size: 20
I1126 07:37:52.814805 14715 communicator.cc:81] communicator_min_send_grad_num_before_recv: 20
I1126 07:37:52.814818 14715 communicator.cc:83] communicator_thread_pool_size: 5
I1126 07:37:52.814831 14715 communicator.cc:85] communicator_send_wait_times: 5
I1126 07:37:52.814843 14715 communicator.cc:87] communicator_max_merge_var_num: 20
I1126 07:37:52.814855 14715 communicator.cc:89] communicator_fake_rpc: 0
I1126 07:37:52.814868 14715 communicator.cc:90] communicator_merge_sparse_grad: 1
I1126 07:37:52.814882 14715 communicator.cc:92] communicator_is_sgd_optimizer: 0
I1126 07:37:52.816067 14715 communicator.cc:330] Communicator start
I1126 07:37:53.000705 14715 rpc_client.h:107] init rpc client with trainer_id 0
2019-11-26 07:37:53,110 - INFO - file list: ['train_data/part-1']
Epoch 0 auc     auc_0.tmp_0             lod: {}
        dim: 1
        layout: NCHW
        dtype: double
        data: [0.493614]

Epoch 0 auc     auc_0.tmp_0             lod: {}
        dim: 1
        layout: NCHW
        dtype: double
        data: [0.511984]

2019-11-26 07:38:28,846 - INFO - epoch 0 finished, use time=35

I1126 07:38:28.847295 14715 communicator.cc:347] Communicator stop
I1126 07:38:28.853050 15143 communicator.cc:266] communicator stopped, recv thread exit
I1126 07:38:28.947477 15142 communicator.cc:251] communicator stopped, send thread exit
I1126 07:38:28.947571 14715 communicator.cc:363] Communicator stop done
2019-11-26 07:38:28,948 - INFO - Distribute Train Success!
```

#
## 分布式训练——同步模式(Sync)
同步模式是分布式训练的一种最基本的运行模式，所有节点保持高度同步，每一个batch的全局参数的更新流程，都需要等待所有节点运行完毕才向后进行，因此，运行的效率取决与全局最慢的节点。若有一个节点发生了故障，则全局训练宣告失败。本示例也提供了PaddlePaddle基于参数服务器模式，运行同步的分布式训练的示例。

### 同步模式数据读取

PaddlePaddle同步模式，目前不支持使用dataset的IO方式，只支持pyreader。pyreader具体使用方法可以查阅[PyReader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/io_cn/PyReader_cn.html#pyreader)，示例代码中，基于pyreader方式数据读取代码位于`py_reader_generator.py`。

在代码中引入Pyreader的方式如下，该部分代码位于`sync_train.py`：
```python
from py_reader_generator import CriteoDataset

def get_pyreader(inputs, params):
    # 请确保每一个训练节点都持有不同的训练文件
    # 当我们用本地多进程模拟分布式时，每个进程需要拿到不同的文件
    # 使用 fleet.split_files 可以便捷的以文件为单位分配训练样本
    file_list = [
        str(params.train_files_path) + "/%s" % x
        for x in os.listdir(params.train_files_path)
    ]
    if not int(params.cloud):
        file_list = fleet.split_files(file_list)
    logger.info("file list: {}".format(file_list))

    # 引入pyreader的数据读取规则
    train_generator = CriteoDataset(params.sparse_feature_dim)
    
    # 利用paddle.batch方法，将pyreader的输出构建为小批量样本，同时shuffle
    train_reader = paddle.batch(paddle.reader.shuffle(
        train_generator.train(file_list, fleet.worker_num(),
                              fleet.worker_index()),
        buf_size=params.batch_size * 100),
                                batch_size=params.batch_size)
    
    # 基于网络的inputs创建py_reader对象
    py_reader = fluid.layers.create_py_reader_by_data(capacity=64,
                                                      feed_list=inputs,
                                                      name='py_reader',
                                                      use_double_buffer=False)
    # 更新inputs，建立与pyreader的联系
    inputs = fluid.layers.read_file(py_reader)

    # 使用py_reader对象装饰IO抛出的数据，灌入网络
    py_reader.decorate_paddle_reader(train_reader)
    return inputs, py_reader
```

### 同步训练策略配置
同步模式的其他分布式启动方法与异步模式一致，不再赘述。下面介绍同步模式与异步模式有区别的地方。

- 首先是分布式训练策略的配置，引入`DistributeTranspilerConfig`后，只需要将`sync_mode`设置为True即可。
    ```python
    strategy = DistributeTranspilerConfig()
    strategy.sync_mode = True
    ```

- 然后是program执行策略`ExecutionStrategy`与program构造策略`BuildStrategy`。我们设置执行时使用多线程模式，线程数与我们指定的cpu核心数相等。同时，开启多线程会默认在本地执行`Reduce`操作，确保各个线程之间的同步。
    ```python
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = int(params.cpu_num)
    build_strategy = fluid.BuildStrategy()
    if int(params.cpu_num) > 1:
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
    ```

- 最后是多线程的数据并行program的配置`CompiledProgram`，我们将刚才配置的执行与构造策略传入，将`fleet.main_program`转化为一个多线程的数据并行program。
    ```python
    compiled_prog = fluid.compiler.CompiledProgram(
            fleet.main_program).with_data_parallel(
                loss_name=avg_cost.name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
    ```

### 使用pyreader进行多轮训练
使用pyreader进行多轮训练时，有一些固有的使用方法，如示例代码所示，我们使用`try & except`捕获异常的方式得到reader读取完数据的信号，使用reset重置reader，以进行下一轮训练的数据读取。
```python
for epoch in range(params.epochs):
    start_time = time.time()

    reader.start()
    batch_id = 0
    try:
        while True:
            loss_val, auc_val, batch_auc_val = exe.run(
                program=compiled_prog,
                fetch_list=[
                    avg_cost.name, auc_var.name, batch_auc_var.name
                ])
            loss_val = np.mean(loss_val)
            auc_val = np.mean(auc_val)
            batch_auc_val = np.mean(batch_auc_val)
            if batch_id % 10 == 0 and batch_id != 0:
                logger.info(
                    "TRAIN --> pass: {} batch: {} loss: {} auc: {}, batch_auc: {}"
                    .format(epoch, batch_id,
                            loss_val / params.batch_size, auc_val,
                            batch_auc_val))
            batch_id += 1
    except fluid.core.EOFException:
        reader.reset()

    end_time = time.time()
    logger.info("epoch %d finished, use time=%d\n" %
                ((epoch), end_time - start_time))
```
执行`exe.run()`时，我们传入的是`CompiledProgram`，同时可以通过加入fetch_list来直接获取我们想要监控的变量。

### 运行：本地模拟分布式
同步模式的运行方式与异步方式相似，同样的有两种方式。

#### 方法 运行`loc_cluster.sh`脚本
运行`local_cluster.sh`脚本，设置启动命令为`sync`：
```bash
sh local_cluster.sh sync
```
使用该脚本开启分布式模拟训练，默认启用2x2的训练模式。Trainer与Pserver的运行日志，存放于`./log/`文件夹，保存的模型位于`./model/`。

使用快速验证数据集，本地模拟同步模式的分布式训练的理想输出为：
> pserver.0.log
```bash
INFO:fluid:file list: ['train_data/part-1']
WARNING:root:paddle.fluid.layers.create_py_reader_by_data() may be deprecated in the near future. Please use paddle.fluid.io.DataLoader.from_generator() instead.
get_pserver_program() is deprecated, call get_pserver_programs() to get pserver main and startup in a single call.
I1128 11:34:50.242866   459 grpc_server.cc:477] Server listening on 127.0.0.1:36011 successful, selected port: 36011
```

> trainer.0.log
```bash
INFO:fluid:file list: ['train_data/part-1']
WARNING:root:paddle.fluid.layers.create_py_reader_by_data() may be deprecated in the near future. Please use paddle.fluid.io.DataLoader.from_generator() instead.
server not ready, wait 3 sec to retry...
not ready endpoints:['127.0.0.1:36012']
I1128 11:34:53.424834 32649 rpc_client.h:107] init rpc client with trainer_id 0
I1128 11:34:53.526729 32649 parallel_executor.cc:423] The Program will be executed on CPU using ParallelExecutor, 2 cards are used, so 2 programs are executed in parallel.
I1128 11:34:53.537334 32649 parallel_executor.cc:287] Inplace strategy is enabled, when build_strategy.enable_inplace = True
I1128 11:34:53.541473 32649 parallel_executor.cc:370] Garbage collection strategy is enabled, when FLAGS_eager_delete_tensor_gb = 0
INFO:fluid:TRAIN --> pass: 0 batch: 10 loss: 0.588123535156 auc: 0.497622251208, batch_auc: 0.496669348982
INFO:fluid:TRAIN --> pass: 0 batch: 20 loss: 0.601480102539 auc: 0.501770208439, batch_auc: 0.520060819177
INFO:fluid:TRAIN --> pass: 0 batch: 30 loss: 0.581234985352 auc: 0.513533941098, batch_auc: 0.552742309157
INFO:fluid:TRAIN --> pass: 0 batch: 40 loss: 0.551335083008 auc: 0.523242733864, batch_auc: 0.586762885637
INFO:fluid:TRAIN --> pass: 0 batch: 50 loss: 0.532891052246 auc: 0.538684471661, batch_auc: 0.617389479234
INFO:fluid:TRAIN --> pass: 0 batch: 60 loss: 0.564157531738 auc: 0.552346798675, batch_auc: 0.628245358534
INFO:fluid:TRAIN --> pass: 0 batch: 70 loss: 0.547578674316 auc: 0.565243961316, batch_auc: 0.651260427476
INFO:fluid:TRAIN --> pass: 0 batch: 80 loss: 0.554214599609 auc: 0.57554000345, batch_auc: 0.648544028986
INFO:fluid:TRAIN --> pass: 0 batch: 90 loss: 0.549561889648 auc: 0.585579565556, batch_auc: 0.660180398731
INFO:fluid:epoch 0 finished, use time=40

INFO:fluid:Distribute Train Success!
```
#
## 分布式训练——模型保存及增量训练
### 单机训练中模型的保存
单机训练，使用`fluid.io.save_inference_model()`或其他接口保存模型，各个接口的联系与区别，可以参考API文档：[模型/变量的保存、载入与增量训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/user_guides/howto/training/save_load_variables.html)


### 分布式训练中模型的保存
分布式训练，推荐使用`fleet.save_persisitables(exe,path)`进行模型的保存，`save_persisitables`不会保存网络的结构，仅保存网络中的长期变量。并且通常而言，仅在0号训练节点上进行模型的保存工作。

推荐的仅保存长期变量的原因是：
1. 分布式训练的program中，有许多仅在分布式训练中才会用到的参数与流程，保存这些步骤，是冗余的，耗费带宽的，且会产生不可预知的风险。
2. 在很多应用场景中，分布式训练出的模型与实际上线的模型不一致，仅使用分布式训练出的参数值，参与其他网络的预测，在这样的场景中，就更无必要保存模型结构了。

> 什么是长期变量？
> 
> 在Paddle Fluid中，模型变量可以分为以下几种类型：
> 
> 1. 模型参数：是深度学习模型中被训练和学习的量。由`fluid.framwork.Parameter()`产生，是`fluid.framework.Variable()`的派生类。
> 2. 长期变量 ：是在整个训练过程中持续存在，不会因为一个迭代结束而销毁的变量，所有的模型参数都是长期变量，但并非所有的长期变量都是模型参数。长期变量通过将`fluid.framework.Varibale()`中的`psersistable`属性设置为`True`来声明。长期变量是模型的核心参数。
> 3. 临时变量：不属于上述两种类别的所有变量都是临时变量，只在一个训练迭代中存在，在每一个迭代结束后，所有的临时变量都会被销毁，然后在下一个迭代开始时，创建新的临时变量。例如输入的训练数据，中间层layer的输出等等。


### 分布式增量训练
Paddle的分布式增量训练也十分易用，代码与上述分布式训练代码保持一致，仅需在Pserver初始化时传入初始化模型的地址，该地址可以位于节点硬盘，亦可传入hadoop集群的存储地址。在训练节点，无需代码改动，在运行`fleet.startup_program`时，会从各个pserver上拉取加载好的参数，覆盖本地参数，实现增量训练。
```python
# 增量训练
if fleet.is_server():
    # 初始化参数服务器节点时，传入模型保存的地址
    fleet.init_server(model_path)
    # 运行参数服务器节点
    fleet.run_server()
elif fleet.is_worker():
    # 训练节点的代码无需更改
    # 在运行fleet.startup_program时，训练节点会从pserver上拉取最新参数
```

#
## 预测——离线单机预测

在我们训练完成后，必然需要在测试集上进行验证模型的泛化性能。单机训练得到的模型必然是可以进行单机预测的，那多机训练得到的模型可以在单机上进行预测吗？答案是肯定的。参考示例代码中的`infer.py`实现CTR-DNN的infer流程，得到离线预测的结果。

### 构建预测网络及加载模型参数
在CTR-DNN的应用中，预测网络与训练网络一致，无需更改，我们使用相同的方式构建`inputs`、`loss`、`auc`。加载参数使用`fluid.io.load_persistables()`接口，从保存好的模型文件夹中加载同名参数。
```python
with fluid.framework.program_guard(test_program, startup_program):
    with fluid.unique_name.guard():
        inputs = ctr_model.input_data(params)
        loss, auc_var, batch_auc_var = ctr_model.net(inputs, params)

        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=inputs, place=place)

        fluid.io.load_persistables(
            executor=exe,
            dirname=model_path,
            main_program=fluid.default_main_program())
```
在进行上述流程时，有一些需要关注的细节：
- 传入的program既不是`default_main_program()`，也不是`fleet.main_program`，而是新建的空的program:
    ```python
    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
   ```
   这是容易理解的，因为在测试时，我们要从零开始，保证预测program的干净，没有其他的影响因素。
-  在创建预测网络时，我们加入了`with fluid.unique_name.guard():`，它的作用是让所有新建的参数的自动编号再次从零开始。Paddle的参数`Variable`以变量名作为区分手段，保证变量名相同，就可以从保存的模型中找到对应参数。
  
    paddle创建的临时变量，编号会自动顺延，如果没有指定变量名，可以观察到这一现象，比如：`fc_1.w_0`->`fc_2.w_0`，想要共享相同的参数，必需要保证编号可以对应。

### 测试数据的读取

测试数据的读取我们使用同步模式中使用过的`pyreader`方法。

### AUC的清零步骤
在训练过程中，为了获得全局auc，我们将auc保存为模型参数，参与长期更新，并在保存模型的过程中被一并保存了下来。在预测时，paddle为了计算预测的全局auc，使用相同的规则创建了同名的auc参数。而我们又在加载模型参数的时候，将训练中的auc加载了进来，如果不在预测前将该值清零，会影响我们的预测值的计算。

以下是将auc中间变量置零操作，`_generated_var_0~3`即为paddle自动创建的auc全局参数。
```python
def set_zero(var_name):
    param = fluid.global_scope().var(var_name).get_tensor()
    param_array = np.zeros(param._get_dims()).astype("int64")
    param.set(param_array, place)

auc_states_names = [
    '_generated_var_0', '_generated_var_1', '_generated_var_2',
    '_generated_var_3'
]
for name in auc_states_names:
    set_zero(name)
```

### 运行Infer
为了快速验证，我们仅取用测试数据集的一个part文件，进行测试。在代码目录下，键入以下命令，进行预测：
```python
python -u infer.py &> test.log &
```
测试结果的日志位于`test.log`，仅训练一个epoch后，在`part-220`上的的理想测试结果为：
```bash
2019-11-26 08:56:19,985 - INFO - Test model model/epoch_0
open file success
2019-11-26 08:56:20,323 - INFO - TEST --> batch: 0 loss: [0.5577456] auc: [0.61541704]
2019-11-26 08:56:37,839 - INFO - TEST --> batch: 100 loss: [0.5395161] auc: [0.6346397]
2019-11-26 08:56:55,189 - INFO - {'loss': 0.5571399, 'auc': array([0.6349838])}
2019-11-26 08:56:55,189 - INFO - Inference complete
```

因为快速验证的训练数据与测试数据极少，同时只训练了一轮，所以远远没有达到收敛，且初始化带有随机性，在您的环境下出现测试结果与示例输出不一致是正常情况。

同时，我们对CTR-DNN模型进行了benchmark的测试。
#
## benchmark
在benchmark中，我们对比了tensorflow和paddlepaddle现有的几种分布式模式在CTR-DNN模型上的效果和速度，其中参与对比的paddle分布式模式有同步、pyreader全异步、dataset全异步，这里pyreader和dataset是paddle支持的两种不同的并行数据读取器，本示例中用到的reader为dataset，pyreader的使用方法可以参考[pyreader使用文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/io_cn/PyReader_cn.html#pyreader)。

benchmark介绍：[CTR(DNN) benchmark on paddle](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/ps/distribute_ctr/paddle)

benchmark相关代码及复现方式见[Fleet Repo](https://github.com/PaddlePaddle/Fleet.git)，路径为Fleet/benchmark/ps/distribute_ctr/paddle/。
