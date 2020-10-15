使用InMemoryDataset/QueueDataset进行训练
========================================

注意
----

本教程目前不支持动态图，仅支持在paddle静态图模式下使用，paddle开启静态图模式

.. code:: python

   paddle.enable_static()

简介
----

为了能高速运行模型的训练，我们使用\ ``InMemoryDataset/QueueDataset``\ API进行高性能的IO，具体介绍可以参考文档\ ``InMemoryDataset``
和 ``QueueDataset``,
以下简称Dataset。Dataset是为多线程及全异步方式量身打造的数据读取方式，每个数据读取线程会与一个训练线程耦合，形成了多生产者-多消费者的模式，会极大的加速我们的模型训练。

本文以训练word2vector模型为例，在训练中引入基于Dataset
API读取训练数据的方式，我们直接加载Fleetx预先定义好的word2vector模型，省去一切前期组网调试阶段，无需变更数据格式，只需在我们原本的\ `训练代码 <https://github.com/PaddlePaddle/FleetX/blob/develop/examples/word2vec_app.py>`__\ 中加入以下内容，便可轻松使用Dataset接口来进行训练。以下是使用Dataset接口一个比较完整的流程：

引入dataset
~~~~~~~~~~~

1. 通过\ ``dataset = paddle.distributed.InMemoryDataset()`` 或者
   ``dataset = paddle.distributed.QueueDataset()``\ 创建一个Dataset对象
2. 指定dataset读取的训练文件的列表， 通过\ ``set_filelist``\ 配置。
3. 通过\ ``dataset.init()`` api
   进行Dataset的初始化配置，\ ``init()``\ 接口接收**kwargs参数,
   详见api文档，列举几个配置的初始化

   a. 将我们定义好的数据输入格式传给Dataset, 通过\ ``use_var``\ 配置。

   b. 指定我们的数据读取方式，由\ ``my_data_generator.py``\ 实现数据读取的规则，后面将会介绍读取规则的实现,
      通过\ ``pipe_command``\ 配置。\ ``pipe_command``\ 是Dataset特有的通过管道来读取训练样本的方式，通过\ ``set_filelist``\ 设置的训练样本文件将被作为管道的输入\ ``cat``\ 到管道中经过用户自定义的\ ``pipe_command``\ 最终输出。

   c. 指定数据读取的batch_size，通过batch_size配置。

   d. 指定数据读取的线程数，一般该线程数和训练线程应保持一致，两者为耦合的关系，通过\ ``thread_num``\ 配置。

.. code:: python

   dataset = paddle.distributed.InMemoryDataset()
   batch_size = config.config["batch_size"]
   thread_num = config.config["thread_num"]
   dataset.init(use_var=model.inputs, pipe_command="python my_data_generator.py", batch_size=batch_size, thread_num=thread_num)
   dataset.set_filelist([config.config["train_files_path"]])

如何指定数据读取规则
~~~~~~~~~~~~~~~~~~~~

在上文我们提到了由\ ``my_data_generator.py``\ 实现具体的数据管道读取规则，那么，怎样为dataset创建数据读取的规则呢？
以下是\ ``my_data_generator.py``\ 的全部代码，具体流程如下： 1.
首先我们需要引入data_generator的类，位于\ ``paddle.distributed.fleet.data_generator``\ 。
2.
声明一些在数据读取中会用到的类和库，如示例代码中的\ ``NumpyRandomInt``\ 、\ ``logger``\ 等。
3.
创建一个子类\ ``Word2VecReader``\ ，继承\ ``fleet.data_generator``\ 的基类，基类有多种选择，如果是多种数据类型混合，并且需要转化为数值进行预处理的，建议使用\ ``MultiSlotDataGenerator``\ ；若已经完成了预处理并保存为数据文件，可以直接以\ ``string``\ 的方式进行读取，使用\ ``MultiSlotStringDataGenerator``\ ，能够进一步加速。在示例代码，我们继承并实现了名为\ ``Word2VecReader``\ 的data_generator子类，使用\ ``MultiSlotDataGenerator``\ 方法。
4.
继承并实现基类中的\ ``generate_sample``\ 函数，逐行读取数据。该函数应返回一个可以迭代的reader方法(带有yield的函数不再是一个普通的函数，而是一个生成器generator，成为了可以迭代的对象，等价于一个数组、链表、文件、字符串etc.)
5.
在这个可以迭代的函数中，如示例代码中的\ ``def nce_reader()``\ ，我们定义数据读取的逻辑。例如对以行为单位的数据进行截取，转换及预处理。

6. 最后，我们需要将数据整理为特定的batch的格式，才能够被dataset正确读取，并灌入的训练的网络中。继承并实现基类中的\ ``generate_batch``\ 函数,
   根据设定的’batch_size’,
   该函数会在\ ``generator_sample``\ 函数产生样本数达到\ ``batch_size``\ 时，调用该函数内队逐条样本的处理逻辑，如示例代码中的\ ``def local_iter()``\ 。
7. 简单来说，数据的输出顺序与我们在网络中创建的\ ``inputs``\ 必须是严格一一对应的，并转换为类似字典的形式。在示例代码中，我们将参数名与数值构成的元组组成了一个list，并将其yield输出。如果展开来看，我们输出的数据形如\ ``[('input_word',[value]),('true_label',[value]),('neg_label',[value])]``

.. code:: python

   import sys
   import io
   import os
   import re
   import collections
   import time
   import config
   import logging

   import paddle
   import numpy as np
   import paddle.distributed.fleet as fleet

   logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
   logger = logging.getLogger("fluid")
   logger.setLevel(logging.INFO)

   class NumpyRandomInt(object):
       def __init__(self, a, b, buf_size=1000):
           self.idx = 0
           self.buffer = np.random.random_integers(a, b, buf_size)
           self.a = a
           self.b = b

       def __call__(self):
           if self.idx == len(self.buffer):
               self.buffer = np.random.random_integers(self.a, self.b,
                                                       len(self.buffer))
               self.idx = 0

           result = self.buffer[self.idx]
           self.idx += 1
           return result

   class Word2VecReader(fleet.MultiSlotDataGenerator):
       def init(self,
                dict_path,
                nce_num,
                window_size=5):
           
           self.window_size_ = window_size
           self.nce_num = nce_num

           word_all_count = 0
           id_counts = []
           word_id = 0

           with io.open(dict_path, 'r', encoding='utf-8') as f:
               for line in f:
                   word, count = line.split()[0], int(line.split()[1])
                   word_id += 1
                   id_counts.append(count)
                   word_all_count += count

           self.word_all_count = word_all_count
           self.corpus_size_ = word_all_count
           self.dict_size = len(id_counts)
           self.id_counts_ = id_counts

           logger.info("corpus_size:", self.corpus_size_)
           self.id_frequencys = [
               float(count) / word_all_count for count in self.id_counts_
           ]
           logger.info("dict_size = " + str(self.dict_size) + " word_all_count = " + str(word_all_count))

           self.random_generator = NumpyRandomInt(1, self.window_size_ + 1)

       def get_context_words(self, words, idx):
           """
           Get the context word list of target word.
           words: the words of the current line
           idx: input word index
           window_size: window size
           """
           target_window = self.random_generator()
           start_point = idx - target_window  # if (idx - target_window) > 0 else 0
           if start_point < 0:
               start_point = 0
           end_point = idx + target_window
           targets = words[start_point:idx] + words[idx + 1:end_point + 1]
           return targets
       
       def generate_batch(self, samples):
           def local_iter():
               np_power = np.power(np.array(self.id_frequencys), 0.75)
               id_frequencys_pow = np_power / np_power.sum()
               cs = np.array(id_frequencys_pow).cumsum()
               result = [[], []]
               for sample in samples:
                   tensor_result = [("input_word", []), ("true_label", []), ("neg_label", [])]
                   tensor_result[0][1].extend(sample[0])
                   tensor_result[1][1].extend(sample[1])
                   neg_array = cs.searchsorted(np.random.sample(self.nce_num))
                   
                   tensor_result[2][1].extend(neg_array)

                   yield tensor_result
           return local_iter
       

       
       def generate_sample(self, line):
           def nce_reader():
               
               word_ids = [int(w) for w in line.split()]
               for idx, target_id in enumerate(word_ids):
                   context_word_ids = self.get_context_words(
                       word_ids, idx)
                   for context_id in context_word_ids:
                       yield [target_id], [context_id]
           return nce_reader

   if __name__ == "__main__":
       my_data_generator = Word2VecReader()
       my_data_generator.init(config.config["dict_path"], config.config["nce_num"])
       my_data_generator.set_batch(config.config["batch_size"])

       my_data_generator.run_from_stdin()

快速调试Dataset
~~~~~~~~~~~~~~~

我们可以脱离组网架构，单独验证Dataset的输出是否符合我们预期。使用命令
``cat 数据文件 | python dataset读取python文件``\ 进行dataset代码的调试：

.. code:: bash

   cat train_data/part_912 | python my_data_generator.py

输出的数据格式如下：
``input_word:size ; input_word:value ; true_label:size ; true_label:value ; neg_label:size ; neg_label:value``

理想的输出为(截取了一个片段)：

.. code:: bash

   ...
   1 112 1 2739 5 6740 451 778 90446 3698
   ...

..

   使用Dataset的一些注意事项 -
   Dataset的基本原理：将数据print到缓存，再由C++端的代码实现读取，因此，我们不能在dataset的读取代码中，加入与数据读取无关的print信息，会导致C++端拿到错误的数据信息。
   -
   dataset目前只支持在\ ``unbuntu``\ 及\ ``CentOS``\ 等标准Linux环境下使用，在\ ``Windows``\ 及\ ``Mac``\ 下使用时，会产生预料之外的错误，请知悉。

数据准备
~~~~~~~~

可以参考\ `文档 <https://github.com/PaddlePaddle/FleetX/tree/fleet_lightning/examples/word2vec>`__
的数据准备部分
完整数据下载以及预处理之后可以选取一个part的文件作为demo数据

.. code:: bash

   mkdir demo_train_data 
   cp train_data/part_1 demo_train_data/

训练
----

我们把原来的\ `训练代码 <(https://github.com/PaddlePaddle/FleetX/blob/develop/examples/word2vec_app.py)>`__:

.. code:: python

   trainer = X.CPUTrainer()
   trainer.fit(model, loader, epoch=10)

替换成如下使用\ ``Dataset``\ 训练的流程, 我们以一个epoch为例：

.. code:: python


   import paddle
   import paddle.fluid as fluid
   import paddle.distributed.fleet as fleet
   import config
   # 开启paddle静态图模式
   paddle.enable_static()

   fleet.init()

   model = X.applications.Word2vec()

   """
   need config loader correctly.
   """

   loader = model.load_dataset_from_file(train_files_path=[config.config["train_files_path"]], dict_path=config.config["dict_path"])

   dist_strategy = fleet.DistributedStrategy()
   dist_strategy.a_sync = True

   optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
   optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
   optimizer.minimize(model.loss)

   if fleet.is_server():
       fleet.init_server()
       fleet.run_server()
   else:
       place = paddle.CPUPlace()
       fleet.init_worker()
       exe = paddle.static.Executor(place)
       default_startup_program = paddle.static.Program()
       default_main_program = paddle.static.Program()
       scope1 = fluid.Scope()
       with fluid.scope_guard(scope1):
           exe.run(model.startup_prog)

       dataset = paddle.distributed.QueueDataset()
       batch_size = config.config["batch_size"]
       thread_num = config.config["thread_num"]
       dataset.init(use_var=model.inputs, pipe_command="python my_data_generator.py", batch_size=batch_size, thread_num=thread_num)
       dataset.set_filelist([config.config["train_files_path"]])

       with fluid.scope_guard(scope1):
           exe.train_from_dataset(model.main_prog, dataset, scope1, debug=False, fetch_list=[model.loss], fetch_info=["loss"], print_period=10)

       fleet.stop_worker()

最后添加上述代码使用的配置文件\ ``config.py``

.. code:: python

   config = dict()

   config["dict_path"] = "thirdparty/test_build_dict"
   config["train_files_path"] = "demo_train_data/part_1"
   config["batch_size"] = 1000
   config["nce_num"] = 5
   config["thread_num"] = 12

通过以上简洁的代码，即可以实现word2vector模型的多线程并发训练
