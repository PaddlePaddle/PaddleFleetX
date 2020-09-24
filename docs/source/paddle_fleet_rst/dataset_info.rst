使用公开数据集开始你的任务
--------------------------

我们在FleetX中为用户提供了数据下载的接口，从HDFS/BOS上下载数据并开始训练。

用户可以使用该接口下载我们提前为您提前准备好的公开数据集（如ImageNet，WiKi等）。

同时也可以下载自己保存在HDFS/BOS上的数据，但保存的数据需要满足特定的格式。

下面我们将为您介绍如何使用该接口下载训练数据，包括接口的使用说明及保存自己数据的方法（以HDFS为例）。

使用说明
~~~~~~~~

利用接口下载数据
^^^^^^^^^^^^^^^^
首先我们需要在yaml文件中配置数据储存的路径（BOS下载时只需要将\ ``data_path``\配置为文件储存的地址）：

.. code:: sh

    # "demo.yaml"
    hadoop_home: ${HADOOP_HOME}
    fs.default.name: ${Your_afs_address}
    hadoop.job.ugi: ${User_ugi_of_your_afs}
    data_path: ${Path_in_afs}


接下来我们可以开始定义训练脚本（"resnet_app.py"）。

在下载之前我们需要引入\ ``fleet``\ 及 \ ``fleetx``\ 模块，并对fleet进行初始化。

.. code:: python

    import paddle
    import paddle.distributed.fleet as fleet
    import fleetx as X

    paddle.enable_static()
    fleet.init(is_collective=True)


对\ ``fleet``\做完初始化后，我们就可以使用\ ``fleetx.Downloader``\下载事先准备好的数据了：

在\ ``download_from_hdfs``\接口中，我们为用户提供了两种下载方式：

- 默认情况下，每台机器会下载全量的数据

- 若在数据并行的场景中，每台机器没有必要储存全量数据。用户可以修改接口中的 \ ``shard_num = fleet.worker_num()``\ 及 \ ``shard_id = fleet.worker_id()``\参数，使得每台机器下载分片的数据。

.. code:: python

    downloader = X.utils.Downloader()
    local_path = downloader.download_from_hdfs('demo.yaml', local_path='.')

下载完数据后即可对模型进行训练：

.. code:: python

    loader = model.get_train_dataloader("{}/train.txt".format(local_path), batch_size=32)
    dist_strategy = fleet.DistributedStrategy()
    optimizer = paddle.fluid.optimizer.Adam(learning_rate=0.001)
    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    optimizer.minimize(cost)
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    trainer = X.MultiGPUTrainer()
    trainer.fit(model, data_loader, epoch=10)

关于FleetX的模型相关实现，请参考：\ `FleetX快速开始 <fleetx_quick_start.html>`__

最后用户可以使用\ `fleetrun <fleetrun_usage_cn.html>`__ 指令开始模型训练：

.. code:: sh

    fleetrun --gpus 0,1,2,3 resnet_app.py


数据储存
^^^^^^^

如上文所说，储存在HDFS/BOS上的数据需要有特定的数据格式，下面我们对数据格式进行详细讲解。

在HDFS/BOS上保存的数据，需要包含以下文件：

.. code:: sh

    .
    |-- filelist.txt
    |-- meta.txt
    |-- train.txt
    |-- val.txt
    |-- a.tar
    |-- b.tar
    |-- c.tar

其中，以\ ``.tar``\结尾的文件为分片保存的数据，全部解压后便可获得全量数据集，一般文件个数为8的倍数。

\ ``filelist.txt``\中记录了所有上述的\ ``.tar``\文件，并记录了每个文件的md5值用于验证是否下载了全量的数据。

可以用\ ``md5sum * | grep ".tar" | awk '{print $2, $1}' > filelist.txt``\命令生成。

在这个例子中\ ``filelist.txt``\为：

.. code:: sh

    a.tar {md5of_a}
    b.tar {md5of_b}
    c.tar {md5of_c}

\ ``meta.txt``\ 中为每台机器中必须下载的文件。有时用户需要每台机器只下载一部分数据，但有些文件需要每台机器都下载，
如：train.txt，val.txt，验证数据集等



\ ``train.txt``\ 及 \ ``val.txt``\中分别记录了训练/数据的数据列表，在训练时dataloader会根据里面的信息读取数据。


BOS数据集
^^^^^^^^

下面是我们为您准备的BOS下载数据配置的地址，用于下载我们在BOS上传的小公开数据集：

- *ImageNet：* https://fleet.bj.bcebos.com/small_datasets/yaml_example/imagenet.yaml

- *Wiki 中文：* https://fleet.bj.bcebos.com/small_datasets/yaml_example/wiki_cn.yaml

- *Wiki 英文：* https://fleet.bj.bcebos.com/small_datasets/yaml_example/wiki_en.yaml
