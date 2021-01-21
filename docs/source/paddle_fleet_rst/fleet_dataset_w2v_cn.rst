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

本文以训练wide&deep模型为例，在训练中引入基于Dataset
以下是使用Dataset接口一个比较完整的流程：

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
   dataset.init(use_var=model.inputs, pipe_command="python reader.py", batch_size=batch_size, thread_num=thread_num)
   dataset.set_filelist([config.config["train_files_path"]])

如何指定数据读取规则
~~~~~~~~~~~~~~~~~~~~

在上文我们提到了由\ ``my_data_generator.py``\ 实现具体的数据管道读取规则，那么，怎样为dataset创建数据读取的规则呢？
以下是\ ``reader.py``\ 的全部代码，具体流程如下： 1.
首先我们需要引入data_generator的类，位于\ ``paddle.distributed.fleet.data_generator``\ 。
2.
声明一些在数据读取中会用到的类和库。
3.
创建一个子类\ ``WideDeepDatasetReader``\ ，继承\ ``fleet.data_generator``\ 的基类，基类有多种选择，如果是多种数据类型混合，并且需要转化为数值进行预处理的，建议使用\ ``MultiSlotDataGenerator``\ ；若已经完成了预处理并保存为数据文件，可以直接以\ ``string``\ 的方式进行读取，使用\ ``MultiSlotStringDataGenerator``\ ，能够进一步加速。在示例代码，我们继承并实现了名为\ ``Word2VecReader``\ 的data_generator子类，使用\ ``MultiSlotDataGenerator``\ 方法。
4.
继承并实现基类中的\ ``generate_sample``\ 函数，逐行读取数据。该函数应返回一个可以迭代的reader方法(带有yield的函数不再是一个普通的函数，而是一个生成器generator，成为了可以迭代的对象，等价于一个数组、链表、文件、字符串etc.)
5.
在这个可以迭代的函数中，如示例代码中的\ ``def wd_reader()``\ ，我们定义数据读取的逻辑。例如对以行为单位的数据进行截取，转换及预处理。

6. 最后，我们需要将数据整理为特定的batch的格式，才能够被dataset正确读取，并灌入的训练的网络中。使用基类中的\ ``generate_batch``\ 函数, 我们无需再做声明
   根据设定的’batch_size’,
   该函数会在\ ``generator_sample``\ 函数产生样本数达到\ ``batch_size``\ 时，调用该函数内队逐条样本的处理逻辑，如示例代码中的\ ``def local_iter()``\ 。
7. 简单来说，数据的输出顺序与我们在网络中创建的\ ``inputs``\ 必须是严格一一对应的，并转换为类似字典的形式。在示例代码中，我们将参数名与数值构成的元组组成了一个list，并将其yield输出。如果展开来看，我们输出的数据形如\ ``[('dense_input',[value]),('C1',[value]),......('label',[value])]``

.. code:: python

    import paddle
    import paddle.distributed.fleet as fleet
    import os
    import sys

    cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
    cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
    hash_dim_ = 1000001
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)


    class WideDeepDatasetReader(fleet.MultiSlotDataGenerator):

        def line_process(self, line):
            features = line.rstrip('\n').split('\t')
            dense_feature = []
            sparse_feature = []
            for idx in continuous_range_:
                if features[idx] == "":
                    dense_feature.append(0.0)
                else:
                    dense_feature.append(
                        (float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1])
            for idx in categorical_range_:
                sparse_feature.append(
                    [hash(str(idx) + features[idx]) % hash_dim_])
            label = [int(features[0])]
            return [dense_feature]+sparse_feature+[label]
        
        def generate_sample(self, line):
            def wd_reader():
                input_data = self.line_process(line)
                feature_name = ["dense_input"]
                for idx in categorical_range_:
                    feature_name.append("C" + str(idx - 13))
                feature_name.append("label")
                yield zip(feature_name, input_data)
            
            return wd_reader

    if __name__ == "__main__":
        my_data_generator = WideDeepDatasetReader()
        my_data_generator.set_batch(16)

        my_data_generator.run_from_stdin()

快速调试Dataset
~~~~~~~~~~~~~~~

我们可以脱离组网架构，单独验证Dataset的输出是否符合我们预期。使用命令
``cat 数据文件 | python dataset读取python文件``\ 进行dataset代码的调试：

.. code:: bash

   cat data/part-0 | python reader.py

输出的数据格式如下：
``13 0.0 0.00663349917081 0.01 0.0 0.0423125 0.054 0.12 0.0 0.074 0.0 0.4 0.0 0.0 1 371155 1 846239 1 204942 1 600511 1 515218 1 906818 1 369888 1 507110 1 27346 1 698085 1 348211 1 170408 1 597913 1 255651 1 415979 1 186815 1 342789 1 994402 1 880474 1 984402 1 208306 1 26235 1 410878 1 701750 1 934391 1 552857 1 1``

理想的输出为(截取了一个片段)：

.. code:: bash

   ...
   13 0.0 0.00663349917081 0.01 0.0 0.0423125 0.054 0.12 0.0 0.074 0.0 0.4 0.0 0.0 1 371155 1 846239 1 204942 1 600511 1 515218 1 906818 1 369888 1 507110 1 27346 1 698085 1 348211 1 170408 1 597913 1 255651 1 415979 1 186815 1 342789 1 994402 1 880474 1 984402 1 208306 1 26235 1 410878 1 701750 1 934391 1 552857 1 1
   ...

..

   使用Dataset的一些注意事项 -
   Dataset的基本原理：将数据print到缓存，再由C++端的代码实现读取，因此，我们不能在dataset的读取代码中，加入与数据读取无关的print信息，会导致C++端拿到错误的数据信息。
   -
   dataset目前只支持在\ ``unbuntu``\ 及\ ``CentOS``\ 等标准Linux环境下使用，在\ ``Windows``\ 及\ ``Mac``\ 下使用时，会产生预料之外的错误，请知悉。

数据准备
~~~~~~~~


完整数据下载以及预处理之后可以选取一个part的文件作为demo数据保存在data目录下


训练
----


.. code:: python


   import paddle
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

   strategy = fleet.DistributedStrategy()
   strategy.a_sync = True
   optimizer = fleet.distributed_optimizer(optimizer, strategy)

   optimizer.minimize(model.cost)

   if fleet.is_server():
       fleet.init_server()
       fleet.run_server()

   if fleet.is_worker():
       place = paddle.CPUPlace()
       exe = paddle.static.Executor(place)

       exe.run(paddle.static.default_startup_program())

       fleet.init_worker()

       distributed_training(exe, model)
       clear_metric_state(model, place)

       fleet.stop_worker()

完整示例代码可以参考 FleetX/examples/wide_and_deep_dataset 目录



通过以上简洁的代码，即可以实现wide&deep模型的多线程并发训练
