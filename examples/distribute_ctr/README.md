# 从零开始的分布式CTR-DNN
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是支持进一步地“商品推送/广告投放”等决策的基础。 简单来说，CTR预估是对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型就是综合考虑各种因素、特征，在大量历史数据上训练得到的模型。

跟随这篇文档，将从零开始，一步步教您如何利用PaddlePaddle Fluid高效且易用的Api，搭建单机ctr-dnn深度学习模型，并利用高阶分布式Api Fleet将其升级为可以在CPU集群中运行的`参数服务器`式分布式深度学习模型。在学习本篇文档后，您可以入门Paddle组网，Paddle分布式模型的搭建，了解CPU多线程全异步模式的启用方法。

## 运行环境检查

- 请确保您的运行环境基于Linux，示例代码同时支持`unbuntu`及`CentOS`
- 请确保您的paddle版本高于`1.6.1`，可以利用pip升级您的paddle版本
- 请确保您的本地模拟分布式运行环境中没有设置`http/https`代理，可以在终端键入`env`查看环境变量

## 代码地址

- 示例代码位于：https://github.com/PaddlePaddle/Fleet/tree/develop/examples/distribute_ctr
  
  在工作环境安装git后，在工作目录克隆Fleet代码仓库，示例代码位于`Fleet/example/distribute_ctr`
- 示例代码结构为：

   ```text
      .
      ├── get_data.sh               # 数据下载脚本
      ├── loc_cluster.sh            # 本地模拟分布式一键启动脚本
      ├── argument.py               # 超参数及环境变量配置
      ├── network.py                # CTR网络结构
      ├── local_train.py            # 单机训练示例代码
      ├── distribute_train.py       # 分布式训练示例代码
      ├── infer.py                  # 模型测试示例代码
      ├── dataset_generator.sh      # dataset数据读取示例代码
      ├── py_reader_generator.py    # pyreader数据读取示例代码
      ├── README.md                 # 使用说明
   ```
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
训练数据放置于`./train_data/`，测试数据放置于`./test_data/`。

至此，我们已完成数据准备的全部工作。

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
将离散数据通过embedding查表得到的值，与连续数据的输入进行`cancat`操作，合为一个整体输入，作为全链接层的原始输入。我们共设计了3层FC，每层FC的输出维度都为400，每层FC都后接一个`relu`激活函数，每层FC的初始化方式为符合正态分布的随机初始化，标准差与上一层的输出维度的平方根成反比。
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

## 数据高速读取
为了能高速运行CTR模型的训练，我们使用`dataset`API进行高性能的IO，dataset是为多线程及全异步方式量身打造的数据读取方式，每个数据读取线程会与一个训练线程耦合，形成了多生产者-多消费者的模式，会极大的加速我们的模型训练。

如何在我们的训练中引入dataset读取方式呢？无需变更数据格式，只需在我们的训练代码中加入以下内容，便可达到媲美二进制读取的高效率，以下是一个比较完整的流程：

### 引入dataset

- 通过工厂类`fluid.DatasetFactory()`创建一个dataset对象。
- 将我们定义好的数据输入格式传给dataset，通过`dataset.set_use_var(inputs)`实现。
- 指定我们的数据读取方式，由`dataset_generator.py`实现数据读取的规则，后面将会介绍读取规则的实现。
- 指定数据读取的batchsize。
- 指定数据读取的线程数，该线程数和训练线程应保持一致，两者为耦合的关系。
- 指定dataset读取的训练文件的列表。
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
- 首先我们需要引入dataset的库，位于`paddle.fluid.incubate.data_generator`。
- 声明一些在数据读取中会用到的变量，如示例代码中的`cont_min_`、`categorical_range_`等。
- 创建一个子类，继承dataset的基类，基类有多种选择，如果是多种数据类型混合，并且需要转化为数值进行预处理的，建议使用`MultiSlotDataGenerator`；若已经完成了预处理，可以直接以`string`的方式进行读取，则可以使用`MultiSlotStringDataGenerator`，能够进一步加速。如示例代码，我们继承并实现了名为`CriteoDataset`的dataset子类。
- 继承并实现基类中的`generate_sample`函数,逐行读取数据。该函数应返回一个可以迭代的reader方法(带有yield的函数不再是一个普通的函数，而是一个生成器generator，成为了可以迭代的对象，等价于一个数组、链表、文件、字符串etc.)
- 在这个可以迭代的函数中，如示例代码中的`def reader()`，我们定义数据读取的逻辑。例如对以行为单位的数据进行截取，转换及预处理。
- 最后，我们需要将数据整理为特定的格式，才能够被dataset正确读取，并灌入的训练的网络中。简单来说，数据的输出顺序与我们在网络中创建的`inputs`必须是严格一一对应的，并转换为类似字典的形式。在示例代码中，我们使用`zip`的方法将参数名与数值构成的元组组成了一个list，并将其yield输出。如果展开来看，我们输出的数据形如`[('dense_feature',[value]),('C1',[value]),('C2',[value]),...,('C26',[value]),('label',[value])]`


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






