# 从零开始的分布式Word2Vec
这篇文档，将从零开始，一步步教您如何从搭建word2vec单机模型，升级为可以在集群中运行的CPU分布式模型。在完成该示例后，您可以入门PaddlePaddle参数服务器搭建，了解CPU多线程全异步模式的启用方法，并能够使用GEO-SGD模式加速分布式的训练。

## 运行环境检查

- 请确保您的运行环境为`Unbuntu`或`CentOS`
- 请确保您的`PaddlePaddle`版本为`1.6.x`
- 请确保您的分布式CPU集群支持运行PaddlePaddle
- 请确保您的运行环境中没有设置`http/https`代理

## 代码下载
本示例代码运行于Linux环境中，请先安装`git`：https://git-scm.com
，然后在工作目录Clone代码仓库，示例代码位于`Fleet/examples/word2vec`：
```bash
git clone https://github.com/PaddlePaddle/Fleet.git
cd Fleet/examples/word2vec
```

## 数据准备
可以使用一键命令进行数据的下载与预处理：
```bash
sh prepare_data.sh
```
也可以跟随下述文档，一步步进行数据的准备工作。

### 训练数据下载
在本示例中，Word2Vec模型使用[1 Billion Word Language Model Benchmark](http://www.statmt.org/lm-benchmark)的训练集，该训练集一共包含30294863个文本，在linux环境下可以执行以下命令进行数据的下载：
```bash
mkdir data
wget --no-check-certificate http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar xzvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
mv 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ data/
```
您也可以从国内源上下载数据，速度更快更稳定。国内源上备用数据下载命令：
```bash
mkdir data
wget --no-check-certificate https://paddlerec.bj.bcebos.com/word2vec/1-billion-word-language-modeling-benchmark-r13output.tar
tar xvf 1-billion-word-language-modeling-benchmark-r13output.tar
mv 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ data/
```
### 预测数据下载
```bash
wget --no-check-certificate https://paddlerec.bj.bcebos.com/word2vec/test_dir.tar
tar -xvf test_dir.tar
mkdir test_data
mv data/test_dir/* ./test_data
```
全量测试集共包含19558个测试样例，每个测试样例由4个词组合构成，依次记为`word_a, word_b, word_c, word_d`。组合中，前两个词`word_a`和`word_b`之间的关系等于后两个词`word_c`和`word_d`之间的关系，例如:
> Beijing China Tokyo Japan  
> write writes go goes

所以word2vec的测试任务实际上是一个常见的词类比任务，我们希望通过公式`emb(word_b) - emb(word_a) + emb(word_c)`计算出的词向量和`emb(word_d)`最相近。最终整个模型的评分用成功预测出`word_d`的数量来衡量。

### 数据预处理
训练集解压后以training-monolingual.tokenized.shuffled目录为预处理目录，预处理主要包括三步，构建词典、数据采样过滤和数据整理。

第一步根据训练语料生成词典，词典格式: 词<空格>词频，出现次数低于5的词被视为低频词，用'UNK'表示：

```bash
python preprocess.py --build_dict --build_dict_corpus_dir data/training-monolingual.tokenized.shuffled --dict_path data/test_build_dict
```
最终得到的词典大小为354051，部分示例如下：
```text
the 41229870
to 18255101
of 17417283
a 16502705
and 16152335
in 14832296
s 7154338
that 6961093
...
<UNK> 2036007
...
```

第二步数据采样过滤，训练语料中某一个单词被保留下来的概率为：$p_{keep} = \frac{word_count}{down_sampling * corpus_size}$。其中$word_count$为单词在训练集中出现的次数，$corpus_size$为训练集大小，$down_sampling$为下采样参数。
```bash
python preprocess.py --filter_corpus --dict_path data/test_build_dict --input_corpus_dir data/training-monolingual.tokenized.shuffled --output_corpus_dir data/convert_text8 --min_count 5 --downsample 0.001
```
与此同时，这一步会将训练文本转成id的形式，保存在data/convert_text8目录下，单词和id的映射文件名为词典+"_word_to_id_"。

最后一步，数据整理。为了方便之后的训练，我们统一将训练数据放在train_data目录下，测试集放在test_data目录下，词表和id映射文件放在thirdparty目录下。同时，为了在多线程分布式训练中达到数据平衡， 从而更好的发挥分布式加速性能，训练集文件个数需尽可能是trainer节点个数和线程数的公倍数，本示例中我们将训练数据重新均匀拆分成1024个文件，您可根据自身情况选择合适的文件个数。
```bash
mkdir thirdparty
mv data/test_build_dict thirdparty/
mv data/test_build_dict_word_to_id_ thirdparty/

python preprocess.py --data_resplit --input_corpus_dir=data/convert_text8 --output_corpus_dir=train_data

mv data/test_dir test_data/
rm -rf data/
```
至此，你已经完成了数据准备阶段的所有步骤。

## 模型
首先让我们从零开始搭建单机word2vec模型。

### 模型设计及代码
本示例实现了基于skip_gram的Word2Vec模型，模型设计可以参考：
>https://aistudio.baidu.com/aistudio/projectDetail/124377

论文可以参考：
>https://arxiv.org/pdf/1301.3781.pdf

基于PaddlePaddle复现的官方Word2Vec模型可以参考：
>https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/word2vec


### 模型组网
#### 输入层
本示例采用经典的负采样+skip_gram模型，所以输入层包括三个，中心词`input_word`，上下文窗口内的正样例`true_word`，负采样得到的若干负样例`neg_word`，其中负样例个数由conf.py中的`neg_num`超参决定。
```python
input_word = fluid.layers.data(name="input_word", shape=[1], dtype='int64',lod_level=1)
true_word = fluid.layers.data(name='true_label', shape=[1], dtype='int64',lod_level=1)
neg_word = fluid.layers.data(name="neg_label", shape=[1], dtype='int64',lod_level=1)
inputs = [input_word, true_word, neg_word]
```
#### 网络层
word2vec组网代码参见`network.py`，请根据代码注释阅读理解word2vec的组网部分，如对网络结构不清楚，可自行查阅论文。
```python
def word2vec_net(dict_size, embedding_size, neg_num):
    init_width = 0.5 / embedding_size
    # 查表获取中心词的词向量
    input_emb = fluid.layers.embedding(
        input=inputs[0],
        is_sparse=True,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb',
            initializer=fluid.initializer.Uniform(-init_width, init_width)))
    # 查表获取正样例对应的权重w
    true_emb_w = fluid.layers.embedding(
        input=inputs[1],
        is_sparse=True,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb_w', initializer=fluid.initializer.Constant(value=0.0)))
    # 查表获取正样例对应的偏置b
    true_emb_b = fluid.layers.embedding(
        input=inputs[1],
        is_sparse=True,
        size=[dict_size, 1],
        param_attr=fluid.ParamAttr(
            name='emb_b', initializer=fluid.initializer.Constant(value=0.0)))

    neg_word_reshape = fluid.layers.reshape(inputs[2], shape=[-1, 1])
    neg_word_reshape.stop_gradient = True
    # 查表获取负样例对应的权重w
    neg_emb_w = fluid.layers.embedding(
        input=neg_word_reshape,
        is_sparse=True,
        size=[dict_size, embedding_size],
        param_attr=fluid.ParamAttr(
            name='emb_w', learning_rate=1.0))
    neg_emb_w_re = fluid.layers.reshape(
        neg_emb_w, shape=[-1, neg_num, embedding_size])
    # 查表获取负样例对应的偏置b
    neg_emb_b = fluid.layers.embedding(
        input=neg_word_reshape,
        is_sparse=True,
        size=[dict_size, 1],
        param_attr=fluid.ParamAttr(
            name='emb_b', learning_rate=1.0))
    neg_emb_b_vec = fluid.layers.reshape(neg_emb_b, shape=[-1, neg_num])
    
    # 计算正样例的分布概率
    true_logits = fluid.layers.elementwise_add(
        fluid.layers.reduce_sum(
            fluid.layers.elementwise_mul(input_emb, true_emb_w),
            dim=1,
            keep_dim=True),
        true_emb_b)

    input_emb_re = fluid.layers.reshape(
        input_emb, shape=[-1, 1, embedding_size])
    neg_matmul = fluid.layers.matmul(
        input_emb_re, neg_emb_w_re, transpose_y=True)
    neg_matmul_re = fluid.layers.reshape(neg_matmul, shape=[-1, neg_num])
    # 计算负样例的分布
    neg_logits = fluid.layers.elementwise_add(neg_matmul_re, neg_emb_b_vec)
    
    label_ones = fluid.layers.fill_constant_batch_size_like(
        true_logits, shape=[-1, 1], value=1.0, dtype='float32')
    label_zeros = fluid.layers.fill_constant_batch_size_like(
        true_logits, shape=[-1, neg_num], value=0.0, dtype='float32')

    # 交叉熵计算loss
    true_xent = fluid.layers.sigmoid_cross_entropy_with_logits(true_logits,
                                                               label_ones)
    neg_xent = fluid.layers.sigmoid_cross_entropy_with_logits(neg_logits,
                                                              label_zeros)
    cost = fluid.layers.elementwise_add(
        fluid.layers.reduce_sum(
            true_xent, dim=1),
        fluid.layers.reduce_sum(
            neg_xent, dim=1))
    avg_cost = fluid.layers.reduce_mean(cost)
    return avg_cost, inputs
```

## 数据读取
在本示例中，数据读取使用[dataset API](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/api_cn/dataset_cn.html)。dataset是一种高性能的IO方式，在分布式应用场景中，多线程全异步模式下，使用dataset进行数据读取加速是最佳选择。

dataset读取数据的代码位于`dataset_generator.py`中，核心部分如下：
```python
def generate_sample(self, line):
    # 数据读取核心代码
    def data_iter():
        cs = np.array(self.id_frequencys_pow).cumsum()
        # 负样例，即neg_word
        neg_array = cs.searchsorted(np.random.sample(neg_num))
        id_ = 0
        word_ids = [w for w in line.split()]
        for idx, target_id in enumerate(word_ids):
            # target_id 当前的中心词，即input_word
            # context_word_dis 上下文窗口中的词，即true_word
            context_word_ids = self.get_context_words(
                word_ids, idx)
            for context_id in context_word_ids:
                neg_id = [ int(str(i)) for i in neg_array ]
                output = [('input_word', [int(target_id)]), ('true_label', [int(context_id)]), ('neg_label', neg_id)]
                yield output
                id_ += 1
                # 每个batch内的neg_array一致，该batch结束后，会重新采样一批负样本
                if id_ % self.batch_size == 0:
                    neg_array = cs.searchsorted(np.random.sample(neg_num)) 
    return data_iter
```

下面简要介绍dataset API的调试方式，在linux环境下，使用以下命令查看运行结果：

```bash
cat data_file | python dataset_generator.py
```

输出的数据格式如下，依次为：
>input_word:size ; input_word:value ; true_word:size ; true_word:value ;neg_word:size ; neg_word:value ;

本示例中，上下文窗口大小为[1,window_size]中随机的一个整数`k`，对于任意一个中心词，我们选择遍历其上下文窗口中的所有单词，依次作为`true_word`，因此最终会得到`2*k`条样本。于此同时，对于一个batch内的样本来说，负样例是一致的。示例输出如下：
```bash
...
# 1代表一个中心词 406为中心词对应id 1代表一个正样例 15代表正样例id 5代表五个负样例 22 851 202 44666 178398依次代表五个负样例对应的id
1 406 1 15 5 22 851 202 44666 178398
1 15 1 406 5 22 851 202 44666 178398
1 15 1 527 5 22 851 202 44666 178398
1 527 1 61 5 22 851 202 44666 178398
1 527 1 220 5 22 851 202 44666 178398
1 527 1 12671 5 22 851 202 44666 178398
1 527 1 4777 5 22 851 202 44666 178398
1 527 1 406 5 22 851 202 44666 178398
1 527 1 15 5 22 851 202 44666 178398
1 527 1 955 5 22 851 202 44666 178398
1 955 1 4777 5 22 851 202 44666 178398
1 955 1 406 5 22 851 202 44666 178398
...
```

如何在模型中引入dataset的数据读取方式呢？

1. 第一步，为dataset设置读取的Variable的格式，在`set_use_var`中添加我们在输入层中设置好的数据，该数据是`list[variable]`的形式；
2. 第二步，我们需要通过`pipe_command`添加读取数据的脚本文件`dataset_generator.py`，dataset类会调用`fluid.DatasetFactory()`其中的`run_from_stdin()`方法进行读取;
3. 第三步，读取过程中的线程数由`set_thread()`方法指定，需要说明的是，利用dataset进行模型训练，读取线程与训练时的线程是耦合的，1个读取队列对应1个训练线程，不同线程持有不同文件，这也就是我们在`数据处理`中强调文件数大于等于线程数的原因所在。
4. 最后，训练数据的batch_size由`set_batch_size()`方法设置。
```python
def get_dataset_reader(inputs):
    dataset = fluid.DatasetFactory().create_dataset()
    # set_use_var的顺序严格要求与读取的顺序一致
    dataset.set_use_var(inputs)
    # 使用pipe command进行数据的高速读取
    pipe_command = "python dataset_generator.py"
    dataset.set_pipe_command(pipe_command)
    dataset.set_batch_size(batch_size)
    thread_num = cpu_num
    # 多线程数据读取可以充分发挥dataset的速度优势
    dataset.set_thread(thread_num)
    return dataset
```

## 单机训练
下面介绍单机训练的方法，单机训练代码见`local_train_example.py`文件。单机训练主要由以下四步组成：

- 网络定义，包括前向、反向及优化网络。

  ```
  # 配置前向网络
  loss, inputs = word2vec_net(dict_size, embedding_size, neg_num)
  # 定义优化器
  optimizer = fluid.optimizer.SGD(
      learning_rate=fluid.layers.exponential_decay(
          learning_rate=learning_rate,
          decay_steps=decay_steps,
          decay_rate=decay_rate,
          staircase=True))
  # 添加反向网络模块，和参数优化更新模块
  optimizer.minimize(loss)
  ```
- reader定义

  ```
  dataset = get_dataset_reader(inputs)
  # 定义训练集
  file_list = [str(train_files_path) + "/%s" % x for x in os.listdir(train_files_path)]
  dataset.set_filelist(file_list)
  ```
- 定义执行器并初始化参数

  ```
  # 定义执行器，并通过run(fluid.default_startup_program())完成参数初始化
  exe = fluid.Executor(fluid.CPUPlace())
  exe.run(fluid.default_startup_program())
  ```
- 执行训练。

  ```python
  for epoch in range(num_epochs):
      dataset.set_filelist(file_list)
      start_time = time.time()
      class fetch_vars(fluid.executor.FetchHandler):
          def handler(self, res_dict):
              loss_value = res_dict['loss']
              logger.info(
                  "epoch -> {}, loss -> {}, at: {}".format(epoch, loss_value, time.ctime()))
      # 开始训练
      exe.train_from_dataset(program=fluid.default_main_program(), dataset=dataset,
                             fetch_handler=fetch_vars(var_dict=var_dict))
      end_time = time.time()
      model_path = str(model_path) + '/trainer_' + str(role.worker_index()) + '_epoch_' + str(epoch)
      # 保存模型
      fluid.io.save_persistables(executor=exe, dirname=model_path)
  logger.info("Train Success!")
  ```
在paddlepaddle中，dataset数据读取是和`train_from_dataset()`训练一一对应的。但是因为dataset设计初衷是保证高速，所以运行于程序底层，与paddlepaddle传统的`feed={dict}`方法不一致，不支持直接通过`train_from_dataset`的返回值监控当前训练的细节，比如loss的变化，但我们可以通过`release/1.6`中新增的`FetchHandler`方法创建一个新的线程，监听训练过程，不影响训练的效率。首先我们需要继承`fluid.executor.FetchHandler`类，获得一个`handler`实例，并用需捕获的变量信息`var_dict`去初始化它。`var_dict`顾名思义，是一个存储待捕获变量的词典，其`key`为字符串类型，主要用来对不同的变量进行区分，可以自由指定，`value`为`Variable`类型。然后，重写`fluid.executor.FetchHandler.handler`函数，监控训练过程。

完成以上步骤，便可以开始进行单机的训练。

## 分布式训练
PaddlePaddle在1.5.0之后新增了高级分布式API`Fleet`，只需数行代码便可将单机模型转换为分布式模型。分布式训练代码见`dist_train_example.py`，单机训练代码见`local_train_example.py`。 这里我们通过代码对比，来说明基于fleet将单机训练转换为分布式训练需要哪些步骤。

1. 引入paddle分布式相关的包
```python
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
```

2. 根据环境变量初始化分布式运行环境。`PaddleCloudRoleMaker`会从当前环境变量中获取所需配置，包括集群server端的地址信息、当前节点的角色是服务器还是训练节点。如果是服务器，则还需给出当前服务器的IP信息和端口信息，如果是训练节点，则还需给出当前节点的trainer_id。具体环境变量的配置可以参考`local_cluster.sh`中给出的示例。随后，我们使用`fleet.init()`方法初始化当前节点的环境配置。
```python
role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)
```

3. 配置不同的分布式策略，然后据此进行切图，插入分布式相关op。分布式训练一般需用户配置的地方仅有两处，第一处是上一步中环境变量的设置，第二处就是对这里的DistributeTranspilerConfig()的配置。`sync_mode`表示是同步训练，还是异步训练。同步训练会给网络中增加`barrier op`来保证各个节点之间的训练速度是一致的，异步训练则没有。于此同时，相较于单机网络，分布式训练中trainer节点的网络会增加`send op`来发送参数梯度信息给pserver。在Fleet API中，为了提高代码的可读性以及简洁性，我们将分布式训练中参数通信的工作都封装到了`padddle.fluid.communicator.Communicator`中，`send op`仅在分布式组网阶段获取通信相关的信息，然后调用`Communicator`中的参数发送单元来实现真正的参数收发。`runtime_split_send_recv`就是用`Communicator`来完成分布式通信的过渡阶段中的一个配置参数，如果设置为True，则代码启用`Communicator`，反之则不启用，依旧通过`send op`来发送参数梯度。此处推荐启用，该参数在将来可能会被废弃。
 
```python
strategy = DistributeTranspilerConfig()
strategy.sync_mode = False
strategy.runtime_split_send_recv = True

# 这一步会调用fluid.transpiler中transpile函数根据当前节点得角色，生成其对应的program
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(loss)
```

4. 启动参数服务器端，这里如果你需要热启，在训练开始之前加载之前某次训练得到的参数，则只需将初始化模型路径传入`init_server()`函数即可。
```python
if role.is_server():
    fleet.init_server()
    fleet.run_server()
```

5. 启动训练节点，训练节点首先调用`init_worker()`来完成节点初始化，然后执行`fleet.startup_program`，从服务器端同步参数的初始化值。接着，和本地训练完全一致，通过执行`fleet.main_program`来完成整个训练过程，并保存模型。一般来说，我们推荐仅让0号训练节点来保存参数即可。同时需着重注意的是，单机使用`fluid.io`接口进行模型的保存，而分布式使用`fleet`接口保存模型。区别在于单机训练仅保存本地内存中的模型参数，而分布式模型保存，全局参数才是准确的训练结果，也就需要从各个Pserver上拉取全局的最新参数进行保存。因此在分布式训练中请使用`fleet`接口保存模型。最后调用`fleet.stop_worker()`关闭训练节点。
```python
elif role.is_worker():
    exe = fluid.Executor(fluid.CPUPlace())
    fleet.init_worker()
    exe.run(fleet.startup_program)

    ## do training like local_train_example.py
    if role.is_first_worker():
        model_path = str(model_path) + '/trainer_' + str(role.worker_index()) + '_epoch_' + str(epoch)
        fleet.save_persistables(executor=exe, dirname=model_path)
    fleet.stop_worker()
```

### Fluid-GEO-SGD模式
PaddlePaddle在1.6.0之后新增了GEO-SGD模式，这种模式也是多线程、全异步、全局无锁的高速模式，支持每个节点在本地训练一定步长后，再与Pserver通信，进行全局参数的更新，可以显著提升训练速度，特别是对于Word2Vec这类有大量稀疏参数的模型，提速会更加明显。开启GEO-SGD模式也是非常的简单，只需在`DistributeTranspilerConfig()`中增加两行配置即可。
```python
strategy = DistributeTranspilerConfig()
strategy.sync_mode = False
strategy.runtime_split_send_recv = True
# 如果开启GEO-SGD模式
if is_geo_sgd:
    strategy.geo_sgd_mode = True
    strategy.geo_sgd_need_push_nums = 400
```
这里`geo_sgd_mode`设为True，代表开启GEO-SGD模式，geo_sgd_need_push_nums设置为400，代表本地训练步长为400，即每个节点只有在完成400个batch的迭代之后才会进行全局通信，完成本地和服务器端的参数更新。

## 训练启动脚本
### 单机训练
```bash
python local_train_example.py
```
### 分布式训练
在示例代码`local_cluster.sh`中，我们配置了一个2x2的参数服务器架构，即两个pserver，两个trainer，
直接执行该脚本即可在本地通过多进程来模拟分布式训练过程：
```bash
sh local_cluster.sh
```

## 模型评估
word2vec模型的评价方法在数据预处理时已介绍基本思路，评估代码见`eval.py`。无论单机训练，还是分布式训练，只需在训练结束后，执行如下命令即可完成评估。
```bash
python eval.py
```

## 调试及优化
`Fleet-ParameterServer-GeoSgd模式`中，我们需要关注两方面的调优：速度与效果。以下参数会影响到GEO-SGD模式的表现。

- 线程数。
  在程序运行时，训练所使用的线程数n，需基于训练节点CPU所具有的核数进行调整。在我们的benchmark测试中，我们设置为16，您可以根据具体环境进行设置。通常来说，线程数越高，训练速度越快。在调整线程数时，您需要关注节点上的文件数是否大于线程数，若文件数少于线程数，则不能如预想的提升速度。我们推荐文件数可以整除线程数，这样可以发挥dataset的最佳性能。

  ```bash
  export CPU_NUM=16
  ``` 
- 本地训练步长。
  训练节点在本地训练的轮数越多，整体通信耗时占比则更低。而频繁的全局参数交互，可以有利于各个节点掌握其他节点参数信息，避免其陷入局部最优。因此，步长`geo_sgd_need_push_num`是一个权衡效果与速度的参数。基于经验值，我们推荐每个线程训练25个batch的步长后进行通信，比如benchmark中，16线程训练，`geo_sgd_need_push_num`可以设置为400。

  ```python
  DistributeTranspilerConfig().geo_sgd_need_push_num = 400
  ```
- 环境变量。
  `FLAGS_communicator_thread_pool_size`环境变量为线程池大小，决定了我们能最多同时启用多少线程来收发参数，增加该值，可以显著提升模型训练速度。但同时，不能够无限增加该变量，这取决于机器的配置与网络带宽。在GEO-SGD模式中，增速的上限为节点数 * 稀疏参数表的数量，超过该值后收益递减。

  ```bash
  export FLAGS_communicator_thread_pool_size=节点数 * 稀疏参数表的数量
  ```
- 全局优化参数的调整。
  由于GEO-SGD是在各个节点上独立训练，使用参数增量进行全局参数的更新，因此一些全局的优化参数需要在该模式下进行调整。比如学习率衰减策略中，我们可以设置`decay_step`来进行学习率的固定步长后衰减，但此处的decay_step为全局参数，本地的样本量不足以迭代预想次数，不能如预期一样衰减到目标学习率。因此，我们需要对decay_step进行同比缩放，除以节点数，从而保证学习率的正常衰减：
  
  ```python
  decay_step = decay_step / trainer_nums
  ```

## benchmark效果
在benchmark中，我们对比了tensorflow和paddlepaddle现有的几种分布式模式在word2vec模型上的效果和速度，其中参与对比的paddle分布式模式有同步、pyreader全异步、dataset全异步、dataset-geo全异步，这里pyreader和dataset是paddle支持的两种不同的并行数据读取器，本示例中用到的reader为dataset，pyreader的使用方法可以参考[pyreader使用文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/io_cn/PyReader_cn.html#pyreader)。

benchmark效果：

benchmark速度：

benchmark相关代码及复现方式见[Fleet Repo](https://github.com/PaddlePaddle/Fleet.git)，路径为Fleet/benchmark/distribute_word2vec/paddle/。
