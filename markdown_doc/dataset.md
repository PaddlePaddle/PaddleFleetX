# 1. 简介

Dataset是Paddle的高性能IO模块。有两种模式InMemory和Queue，并且支持用户通过pipe command自定义数据处理逻辑。

对于计算量较小的网络结构，QueueDataset的时间难以被计算Overlap，小数据下Offline多Epoch训练推荐采用InMemory模式。对于计算复杂型网络结构，推荐使用QueueDataset。

## 1.1 Dataset与Pyreader的加速能力对比

Profile方法：只读数据不训练，读完即消费完。这种方法得到的吞吐数据是IO的极限能力。

以下是IMDB数据不同线程数下的throughputs：

<p align="center">
<img align="center" src="images/dataset_throughputs.png" height="270px" width="940px">
<p>

# 2. 快速开始
下面介绍单机示例，用户可以直接跑通。

```python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import sys
import paddle
import paddle.fluid as fluid

# 假如我们有两个数据文件如下：
with open("a.txt", "w") as f:
    f.write("1 1 2 8 9 4 15 25 35 45 1 21\n")
    f.write("1 2 2 8 10 4 16 26 36 46 1 22\n")
    f.write("1 3 2 9 10 4 17 27 37 47 1 23")
with open("b.txt", "w") as f:
    f.write("1 4 2 8 11 4 15 25 36 45 1 24\n")
    f.write("1 5 2 10 12 4 26 36 46 56 1 85\n")
    f.write("1 6 2 12 13 4 17 27 37 67 1 86\n")
    f.write("1 7 2 9 11 4 18 28 38 48 1 87")

# 定义网络，本示例只定义了数据层
slots = ["slot1","slot2","slot3","slot4"]
slots_vars = []
for slot in slots:
    var = fluid.layers.data(name=slot, shape=[1], dtype="int64", lod_level=1)
    slots_vars.append(var)
    #fluid.layers.Print(var)
# 创建Dataset，并配置
dataset = fluid.DatasetFactory().create_dataset()
dataset.set_batch_size(32)
dataset.set_thread(3)
dataset.set_filelist(["a.txt", "b.txt"])
dataset.set_pipe_command("cat")
dataset.set_use_var(slots_vars)
# 跑训练
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
for i in range(2):
    exe.train_from_dataset(fluid.default_main_program(), dataset)
    print "Run One Epoch Done"
```
执行如下命令：

```
python train.py
```

# 3. API介绍
Dataset是一个工厂类，有如下两种：

（1）QueueDataset：每轮训练从指定的文件列表里读取数据。

（2）InMemoryDataset：用户可以调用LoadIntoMemory把数据读到内存。每轮训练从内存中读取数据。

用户创建Dataset时，默认是QueueDataset，也可以通过如下方法指定

```python
dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
# 或者
# dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
```

## 3.1 API列表速查

| api     |  说明 | 参数类型 | 注意事项 |
| ------------- | ------------- | ------------- | ------------- |
| set_pipe_command (pipe_command)     | 设置pipe command | 字符串      |  |
|set_batch_size (batch_size) | 设置 batch size | int值| |
|set_thread (thread_num)|设置线程数|int值   | |
|set_filelist (filelist)|设置文件列表|list|如果是hdfs或者afs路径，需要以"hdfs:或者"“afs:” 开头。是具体的文件而非目录。|
|set_use_var (var_list)|设置用了哪些var|list|列表中的元素是Variable类型。|
|set_hdfs_config (fs_name, fs_ugi)|设置数据路径的fs name 和 ugi|字符串|如果是本地文件，无需设置。|
|load_into_memory ()|把数据读到内存 |无|只有InMemoryDataset才有。|
|local_shuffle ()|本地shuffle|无||
|global_shuffle (fleet=None, thread_num=12)|全局shuffle|Fleet类|若要全局shuffle，只需global_shuffle，无需再调用local_shuffle。线程数默认12。|
|release_memory ()|清空内存中的数据|无|只有InMemoryDataset才有。|
|set_queue_num(queue_num)|设置数据queue的个数|int值|只有InMemoryDataset才有。|
|set_parse_ins_id(parse_ins_id)|是否解析样本id，默认false|bool|如果设为true，pipe command中的data generator需要解析出样本id。用处主要是dump中间结果时，带上样本id。|
|preload_into_memory(thread_num)|预加载数据|int值|只有InMemoryDataset才有。|
|wait_preload_done|等待预加载数据结束||只有InMemoryDataset才有。|


## 3.2  API详细说明

 - set_pipe_command

    pipe command是对原始的数据做处理，生成Paddle可以识别的格式。Dataset读取的每一行原始数据，都会用这里的命令做处理。

    可以是一个执行python脚本或者二进制等任意Linux命令，用户可以写自己的逻辑。

    pipe command生成var的顺序需要和set_user_var保持一致。

    用法如下：

    ```python
    dataset.set_pipe_command("python my_data_generator.py")
    ```
    
    注意你的python路径，例如
    ```python
    dataset.set_pipe_command("/your/path/to/python/bin/python my_data_generator.py")
    ```

 - set_batch_size

    设置训练的batch size

    用法如下：
    ```python
    dataset.set_batch_size(32)
    ```
    
 - set_thread

    设置Dataset读数据的线程数。如果用户指定的filelist中文件数目少于线程数，则实际创建线程数等于文件数。

    训练线程数与Dataset读数据的线程数是默认相等的。

    用法如下：
    ```python
    dataset.set_thread(12)
    ```

 - set_filelist

    设置要读取的文件列表。在分布式作业的情况下，用户需要自行切分文件列表，set_filelist传入本节点对应的文件列表。

    例如一共有文件：[“1.txt”,  “2.txt”, “3.txt”, “4.txt”, “5.txt”] ，worker节点数为3，那么用户需要先把文件列表切分为三份：(可以使用fleet.split_files切分文件)

    ```python
    if my_rank == 0:
        dataset.set_filelist([“1.txt”, “2.txt”])
    elif my_rank == 1:
        dataset.set_filelist([“3.txt”, “4.txt”])
    else:
        dataset.set_filelist([“5.txt”])
    ```

 - set_use_var

    设置要用到哪些variable，顺序跟经过pipe command处理后生成的各个slot顺序保持一致。

    用法如下：
    ```python
    # 定义网络，本示例只列出了输入层
    data = fluid.layers.data(name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    # ...
    # 相应的，pipe command生成的数据也要是先label后data
    dataset.set_use_var([label，data])
    ```

 - set_hdfs_config

    如果用户set_filelist时，指定的是hdfs或者afs路径，那么需要设置fs name和 fs ugi

    用法如下：
    ```python
    # 假如设置了hdfs路径
    dataset.set_filelist(["hdfs:/app/mpi/1.txt", "hdfs:/app/mpi/2.txt"])
    # ... 
    dataset.set_hdfs_config("my_fs_name", "my_fs_ugi")
    ```

 - load_into_memory

    把数据load到内存中，当该函数执行完，数据已经load到内存中（后面也会支持异步）。

    只有InMemoryDataset才有该方法。

    用法如下：
    ```python
    dataset.load_into_memory()
    ```


 - local_shuffle

    本地shuffle数据，每个节点shuffle自己的数据。

    目前只有InMemoryDataset有该方法，后面QueueDataset也会支持。

    用法如下：
    ```python
    dataset.local_shuffle()
    ```

 - global_shuffle

    全局shuffle所有节点的数据。执行函数后，无需再执行一遍local_shuffle。

    目前只有InMemoryDataset有该方法，后面QueueDataset也会支持。

    用法如下：
    ```python
    dataset.global_shuffle(fleet)
    ```


 - release_memory

    每当executor run完后，用户若想重新设置filelist，那么需要先释放之前的内存中的数据。

    只有InMemoryDataset有该方法。

    用法如下：
    ```python
    dataset.release_memory()
    ```

 - set_queue_num

    设置数据queue的个数，可以与训练线程数不一致。若与与训练线程数不一致，最好能整除。如果不设置，默认就是dataset的线程数。
    
    只有InMemoryDataset有该方法。

    用法如下：
    ```python
    dataset.set_queue_num(12)
    ```
    
 - set_parse_ins_id

    是否解析样本id，默认false如果设为true，pipe command中的data generator需要解析出样本id。

    用处主要是dump中间结果时，带上样本id。

    只有InMemoryDataset有该方法。
    
    用法如下
    ```python
    dataset.set_parse_ins_id(True)
    ```

 - preload_into_memory

    预加载数据，线程数可配，只有InMemoryDataset有该方法。

    用法如下
    ```python
    dataset.preload_into_memory(12)
    ```

 - wait_preload_done

    等待预加载数据结束，只有InMemoryDataset有该方法。

    用法如下
    ```python
    dataset.wait_preload_done(12)
    ```

# 4. 如何在Pipe Command中处理数据

paddle中提供了基类paddle.fluid.incubate.data_generator.MultiSlotStringDataGenerator，用户可以继承并实现自己的处理数据的逻辑。

生成的数据需要与dataset.set_use_var中设置的顺序保持一致。

```python
# -*- coding: UTF-8 -*-
import sys
import os
import paddle
import re
import collections
import time
import paddle.fluid.incubate.data_generator as dg

class MyDataset(dg.MultiSlotStringDataGenerator):
    # 用户可以实现自己的任意逻辑，比如加一个函数load_resource读取一个配置文件
    def load_resource(self, dictf):
        self._all_slots_dict = collections.OrderedDict()
        with open(dictf, 'r') as f:
            slots = f.readlines()
        for index, slot in enumerate(slots):
            self._all_slots_dict[slot.strip()] = [False, index + 2]

    # 处理每一行数据，这个函数是重载基类的。
    def generate_sample(self, line):
        def data_iter():
            # 输入的格式，比如这里是
            # ins_id show click feasign1:slot1 feasign2:slot2 feasign3:slot3 ... \t xxxx \t xxxx ...
            elements = line.split('\t')[0].split()[1:]
            padding = "0"
            # 输出的格式，对应上面的输入，必须是如下格式
            # [ (show, [xxx1]), (click, [xxx2]), (slot1,[feasign1,feasign2,..]), (slot2,[feasign4,feasign5..]), ... ]
            # 需要注意的是，上面的show、click和feasign仅仅是示例，只是为了说明格式。
            # 这些字段需跟set_use_var一致。
            output = [("show", [elements[0]]), ("click", [elements[1]])]
            output += [(slot, []) for slot in self._all_slots_dict]
            for elem in elements[2:]:
                feasign, slot = elem.split(':')
                if not self._all_slots_dict.has_key(slot):
                    continue
                self._all_slots_dict[slot][0] = True
                index = self._all_slots_dict[slot][1]
                output[index][1].append(feasign)
            for slot in self._all_slots_dict:
                visit, index = self._all_slots_dict[slot]
                if visit:
                    self._all_slots_dict[slot][0] = False
                else:
                    output[index][1].append(padding)
            # yield 处理好的数据
            yield output
        return data_iter

d = MyDataset()
d.load_resource("all_slot.dict")
d.run_from_stdin()
相应的，用户可能是如下代码：

show = fluid.layers.data(name="show", shape=[-1, 1], dtype="int64", lod_level=0, append_batch_size=False)
label = fluid.layers.data(name="click", shape=[-1, 1], dtype="int64", lod_level=0, append_batch_size=False)
all_my_slots_var_list = []
for slot in all_my_slots:
    all_my_slots_var_list.append(fluid.layers.data(name=slot, shape=[1], dtype="int64", lod_level=1))
 
dataset = fluid.DatasetFactory().create_dataset(“InMemoryDataset”)
dataset.set_use_var([show, label] + all_my_slots_var_list)
dataset.set_pipe_comamnd("python my_data_generator.py")
.....
```

如果觉得python脚本处理数据较慢，可以用c++自行实现同样的逻辑。

Tips:  (1) fgets、fprintf性能更好   (2) 编译时可以开-O3


data generator最终输出到stdout，c++端读取的格式是feasign个数n + 该slot的n个feasign。

假如一共有两个slot，第一个slot有2个feasign（12345和23456），第二个slot有3个feasign（34567、78999、89999），那么输出的样本如下：

```
2 12345 23456 3 34567 78999 89999
```


# 5. 更多示例

## 5.1 InMemoryDataset

```python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import sys
import paddle
import paddle.fluid as fluid
# 定义网络，本示例只定义了数据层
slots = ["slot1","slot2","slot3","slot4"]
slots_vars = []
for slot in slots:
    var = fluid.layers.data(name=slot, shape=[1], dtype="int64", lod_level=1)
    slots_vars.append(var)
    #fluid.layers.Print(var)
# 创建InMemoryDataset，并配置
dataset = fluid.DatasetFactory().create_dataset(“InMemoryDataset”)
dataset.set_batch_size(32)
dataset.set_thread(3)
dataset.set_filelist(["a.txt", "b.txt"])
dataset.set_pipe_command("cat")
dataset.set_use_var(slots_vars)
# 把数据读到内存
dataset.load_into_memory()
# 本地shuffle
dataset.local_shuffle()
# 跑训练
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
for i in range(2):
    exe.train_from_dataset(fluid.default_main_program(), dataset)
    print "Run One Epoch Done"
```

## 5.2 InMemoryDataset，多次设置filelist 

```python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import sys
import paddle
import paddle.fluid as fluid
# 定义网络，本示例只定义了数据层
slots = ["slot1","slot2","slot3","slot4"]
slots_vars = []
for slot in slots:
    var = fluid.layers.data(name=slot, shape=[1], dtype="int64", lod_level=1)
    slots_vars.append(var)
    #fluid.layers.Print(var)
# 创建Dataset，并配置
dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")

dataset.set_pipe_command("cat")
dataset.set_use_var(slots_vars)

# 跑训练
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
my_batch_size = [16, 32, 64, 128, 256]
my_thread_num = [1, 2, 3, 4, 5]
for i in range(5):
    dataset.set_batch_size(my_batch_size[i])
    dataset.set_thread(my_thread_num[i])
    dataset.set_filelist(["a.txt"] * i)
    dataset.load_into_memory()
    dataset.local_shuffle()
    exe.train_from_dataset(fluid.default_main_program(), dataset)
    dataset.release_memory()
    print "Run One Epoch Done"
```    

## 5.3 QueueDataset，多次设置filelist

```python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import sys
import paddle
import paddle.fluid as fluid
# 定义网络，本示例只定义了数据层
slots = ["slot1","slot2","slot3","slot4"]
slots_vars = []
for slot in slots:
    var = fluid.layers.data(name=slot, shape=[1], dtype="int64", lod_level=1)
    slots_vars.append(var)
    #fluid.layers.Print(var)
# 创建QueueDataset，并配置
dataset = fluid.DatasetFactory().create_dataset("QueueDataset")

dataset.set_pipe_command("cat")
dataset.set_use_var(slots_vars)

# 跑训练
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
my_batch_size = [16, 32, 64, 128, 256]
my_thread_num = [1, 2, 3, 4, 5]
for i in range(5):
    dataset.set_batch_size(my_batch_size[i])
    dataset.set_thread(my_thread_num[i])
    dataset.set_filelist(["a.txt"] * i)
    exe.train_from_dataset(fluid.default_main_program(), dataset)
    print "Run One Epoch Done"
```    
    
## 5.4 preload

对于InMemoryDataset，如果使用多个dataset，如下load数据和训练是串行的：

```python
# -*- coding: UTF-8 -*-
import paddle
import paddle.fluid as fluid
 
slot_1 = fluid.layers.data(name="s1", shape=[1], dtype="int64", lod_level=1)
slot_2 = fluid.layers.data(name="s2", shape=[1], dtype="int64", lod_level=1)
 
def create_demo_dataset():
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    dataset.set_use_var([slot_1, slot_2])
    dataset.set_batch_size(1)
    dataset.set_thread(1)
    dataset.set_pipe_command("cat")
    dataset.set_filelist(["a.txt", "b.txt"])
    return dataset
 
exe = fluid.Executor(fluid.CPUPlace())
 
dataset1 = create_demo_dataset()
dataset2 = create_demo_dataset()
dataset3 = create_demo_dataset()
 
dataset1.load_into_memory()
dataset1.local_shuffle()
exe.train_from_dataset(fluid.default_main_program(), dataset1, fluid.global_scope())
 
dataset2.load_into_memory()
dataset2.local_shuffle()
exe.train_from_dataset(fluid.default_main_program(), dataset2, fluid.global_scope())
 
dataset3.load_into_memory()
dataset3.local_shuffle()
exe.train_from_dataset(fluid.default_main_program(), dataset3, fluid.global_scope())
```

如果想让load数据和训练的时间overlap，可以使用dataset的preload功能：

```python
# -*- coding: UTF-8 -*-
import paddle
import paddle.fluid as fluid
 
slot_1 = fluid.layers.data(name="s1", shape=[1], dtype="int64", lod_level=1)
slot_2 = fluid.layers.data(name="s2", shape=[1], dtype="int64", lod_level=1)
 
def create_demo_dataset():
    dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
    dataset.set_use_var([slot_1, slot_2])
    dataset.set_batch_size(1)
    dataset.set_thread(1)
    dataset.set_pipe_command("cat")
    dataset.set_filelist(["a.txt", "b.txt"])
    return dataset
 
exe = fluid.Executor(fluid.CPUPlace())
 
dataset1 = create_demo_dataset()
dataset2 = create_demo_dataset()
dataset3 = create_demo_dataset()
 
dataset1.load_into_memory()
# 可以开始让dataset2异步读数据了
dataset2.preload_into_memory()
dataset1.local_shuffle()
exe.train_from_dataset(fluid.default_main_program(), dataset1, fluid.global_scope())
 
# 等dataset2异步读数据完成后
dataset2.wait_preload_done()
# 可以开始让dataset3异步读数据了
dataset3.preload_into_memory()
dataset2.local_shuffle()
exe.train_from_dataset(fluid.default_main_program(), dataset2, fluid.global_scope())
 
# 等dataset3异步读数据完成后
dataset3.wait_preload_done()
dataset3.local_shuffle()
exe.train_from_dataset(fluid.default_main_program(), dataset3, fluid.global_scope())
```
