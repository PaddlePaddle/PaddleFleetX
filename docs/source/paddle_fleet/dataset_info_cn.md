## 使用公开数据集开始你的任务

考虑到大多数并行训练的场景都是采用数据并行的方式，FleetX提供了分布式训练场景中对数据操作的基本功能。

- 分片并发下载：
FleetX中为用户提供了数据分片下载的功能，可以将文件系统（HDFS/BOS）中按照一定规则保存的数据分片并发下载。

- 预置公开数据集样例：
FleetX提供了公网可以下载标准数据集子集的能力，方便用户快速获取和使用（如ImageNet，WiKiPedia等）。

- 用户自定义数据集：
FleetX提供能够分片并发下载的数据集具有特定保存格式，用户可以将自己的数据保存为FleetX提供的格式自己使用

### 使用说明

#### 从百度云BOS文件系统获取数据

- demo.yaml
  ``` yaml
  bos_path: https://fleet.bj.bcebos.com/small_datasets/imagenet
  ```

#### 从HDFS文件系统获取数据

- demo.yaml
  ``` sh

  hadoop_home: ${HADOOP_HOME}
  fs.default.name: ${Your_afs_address}
  hadoop.job.ugi: ${User_ugi_of_your_afs}
  data_path: ${Path_in_afs}

  ```

#### 单进程并发下载数据

  ``` python

      import fleetx as X

      downloader = X.utils.Downloader()
      # or
      # local_path = downloader.download_from_bos('demo.yaml', local_path="data")
      local_path = downloader.download_from_hdfs('demo.yaml', local_path="data")
      model = X.applications.Resnet50()
      loader = model.get_train_dataloader(local_path)

  ```

#### 多进程分片并发下载数据

  ``` python

      import paddle.distributed.fleet as fleet
      import fleetx as X

      fleet.init(is_collective=True)

      downloader = X.utils.Downloader()
      # or
      # local_path = downloader.download_from_bos('demo.yaml', local_path="data")
      local_path = downloader.download_from_hdfs(
                        'demo.yaml', local_path="data",
                        shard_num=fleet.worker_num(),
                        shard_id=fleet.worker_index())
      model = X.applications.Resnet50()
      loader = model.get_train_dataloader(local_path)

   ```

  多进程分片下载通常会使用到`paddle.distributed.fleet` API, 通过配置`shard_num`即总分片的数量以及`shard_id`即分片的编号来实现多进程分片下载。在单机就可以验证的例子，通过使用Paddle提供的多进程训练的启动命令`fleetrun --gpus 0,1,2,3 resnet.py`来实现数据分片并发下载。


### 数据存储

如上文所说，储存在HDFS/BOS上的数据需要有特定的数据格式，下面我们对数据格式进行详细讲解。

在HDFS/BOS上保存的数据，需要包含以下文件：

``` sh

    .
    |-- filelist.txt
    |-- meta.txt
    |-- train.txt
    |-- val.txt
    |-- a.tar
    |-- b.tar
    |-- c.tar

```

其中，以`tar`结尾的文件是提前保存好的分片数据，数据本身的格式不做限制，只要具体模型的数据读取器能够读取即可。在这里，我们建议分片的文件数量适合并发下载，既不要非常碎片化也不需要用极
少的文件保存，单个tar文件控制在400M以内即可。

`filelist.txt`中记录了所有上述的`.tar`文件，并记录了每个文件的md5sum值用于在FleetX内部验证是否下载了全量数据。

获取每个tar文件的md5sum可以通过`md5sum * | grep ".tar" | awk '{print $2, $1}' > filelist.txt`命令生成。

在这个例子中`filelist.txt`为：

``` sh

    a.tar {md5of_a}
    b.tar {md5of_b}
    c.tar {md5of_c}

```

考虑到不同的数据集可能有不同的统计信息文件，例如自然语言处理任务中经常使用的词典，我们设计`meta.txt`文件，用来记录整个数据集在每个节点实例上都会下载的文件，比如训练文件列表`train.txt`，验证数据文件列表`val.txt`等


### 预置数据集整体信息

|  数据集来源 | 数据集大小 | BOS提供子集大小 | BOS数据集下载地址 |
|  ----  | ----  | ---- | ---- |
|  [ImageNet](http://www.image-net.org/) | 128万图片 | 6万图片 | [Sample Imagenet](https://fleet.bj.bcebos.com/test/loader/small_imagenet.yaml) |
|  [Wikipedia-En]() | 60,173,276 句对 | 50,412 句对 | [Sample Wiki-En](https://fleet.bj.bcebos.com/test/loader/wiki_en_small.yaml) |
| [Wikipedia-Zh]() | - | 10,958 句对 | [Sample Wiki-Cn](https://fleet.bj.bcebos.com/test/loader/wiki_cn_small.yaml) |
