fleetrun 启动分布式任务
=======================

Paddle提供\ ``fleetrun``\ 命令，只需一行简单的启动命令，即可轻松地将Paddle
Fleet GPU单机单卡任务切换为多机多卡任务，也可将参数服务器单节点任务切换为多个服务节点、多个训练节点的分布式任务。
\ ``fleetrun``\ 在静态图和动态图场景下均可使用。

内容导航
--------
1. 使用要求_
2. 使用说明_

    2.1. GPU场景_
    2.2. CPU场景_

3. fleetrun命令参数介绍_
4. 使用fleetrun进行GPU多卡训练实例_

.. _使用要求:

使用要求
--------

使用\ ``fleetrun``\ 命令的要求:

- 安装 paddlepaddle 2.0-rc 及以上

.. _使用说明:

使用说明
--------

.. _GPU场景:

GPU场景
^^^^^^^

-  **GPU单机单卡训练**

单机单卡有两种方式：一种可直接使用\ ``python``\ 执行，也可以使用\ ``fleetrun``\ 执行。推荐使用\ ``fleetrun``\  启动方法。

【方法一】直接使用\ ``python``\ 执行

.. code:: sh

    export CUDA_VISIBLE_DEVICES=0
    python train.py

【方法二】使用\ ``fleetrun``\ 执行

::

    fleetrun --gpus=0 train.py

注：如果指定了\ ``export CUDA_VISIBLE_DEVICES=0`` ，则可以直接使用：

.. code:: sh

   export CUDA_VISIBLE_DEVICES=0
   fleetrun train.py

-  **GPU单机多卡训练**

若启动单机4卡的任务，只需通过\ ``--gpus``\ 指定4张卡即可。
::

   fleetrun --gpus=0,1,2,3 train.py

注：如果指定了\ ``export CUDA_VISIBLE_DEVICES=0,1,2,3``
，则可以直接使用：

.. code:: sh

   export CUDA_VISIBLE_DEVICES=0,1,2,3
   fleetrun train.py

-  **GPU多机多卡训练**

**[示例一]** 2机8卡 (每个节点4卡)

.. code:: sh

   fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus=0,1,2,3 train.py

注：如果每台机器均指定了\ ``export CUDA_VISIBLE_DEVICES=0,1,2,3``
，则可以直接在每台节点上启动：

.. code:: sh

   export CUDA_VISIBLE_DEVICES=0,1,2,3
   fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" train.py

**[示例二]** 2机16卡（每个节点8卡，假设每台机器均有8卡可使用）

.. code:: sh

   fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" train.py

-  **GPU 在PaddleCloud上提交任务**

**PaddleCloud**\ 是百度开源的云上任务提交工具，提供云端训练资源，打通⽤户云端资源账号，并且支持以命令行形式进行任务提交、查看、终止等多种功能。PaddleCloud更多详情：\ `PaddleCloud <https://github.com/PaddlePaddle/PaddleCloud>`__

百度内部用户在PaddleCloud上启动分布式任务十分方便，执行PaddleCloud启动任务时指定任务所需机器数和卡数，由\ ``-—k8s-trainers``\ 和 \ ``—-k8s-gpu-cards``\ 决定。无论执行单机单卡还是多机多卡任务，只需在提交任务的运行脚本中使用：

.. code:: sh

   fleetrun train.py

使用开源版本的PaddleCloud启动分布式任务时，可以通过\ ``instance_count``\ 指定申请计算节点数目, \ ``instance_count = 1``\ 时默认启动单机任务，\ ``instance_count > 1``\ 时可启动多机任务

.. code:: sh

   paddlecloud submit_job --public_bos=0 --instance_count=2 --bos_url={bucket}.bj.bcebos.com/your/dir --start_cmd="sh run.sh"

在\ ``run.sh``\ 运行脚本中使用fleetrun即可：

.. code:: sh

   fleetrun train.py

.. _CPU场景:

CPU场景
^^^^^^^

-  **单机训练（0个服务节点，1个训练节点）**

Fleet支持参数服务器任务多机回退到单机任务，直接运行时程序将转换为一般的Paddle单机任务。
.. code:: sh

   python train.py

-  **参数服务器训练 - 单机模拟分布式训练（1个服务节点，4个训练节点）**

fleetrun启动时只指定服务节点数\ ``server_num``\ 和 训练节点数\ ``worker_num``\ ，即可进行本地模拟分布式训练，推荐使用此方法进行本地调试。
.. code:: sh

   fleetrun --server_num=1 --worker_num=4 train.py

-  **参数服务器训练 -
   多机训练（2台节点，每台节点均有1个服务节点，4个训练节点）**

fleetrun启动时只指定服务节点的ip和端口列表\ ``servers``\ 和 训练节点的ip和端口列表列表\ ``workers``\ ，即可进行多机训练。
下列示例中，xx.xx.xx.xx代表机器1，yy.yy.yy.yy代表机器2，6170代表随机指定的服务节点的端口。fleetrun将分别在2台机器上启动1个服务节点，4个训练节点。

.. code:: sh
    # 2个servers 8个workers
    fleetrun --servers="xx.xx.xx.xx:6170,yy.yy.yy.yy:6171" --workers="xx.xx.xx.xx:6172,xx.xx.xx.xx:6173,xx.xx.xx.xx:6174,xx.xx.xx.xx:6175,yy.yy.yy.yy:6176,yy.yy.yy.yy:6177,yy.yy.yy.yy:6178,yy.yy.yy.yy:6179" train.py

训练节点 \ ``workers``\ 的端口可以在启动时省略，此时fleetrun将会在启动训练任务前分配好端口给每个训练节点。
.. code:: sh
    # 2个servers 8个workers
    fleetrun --servers="xx.xx.xx.xx:6170,yy.yy.yy.yy:6171" --workers="xx.xx.xx.xx,xx.xx.xx.xx,xx.xx.xx.xx,xx.xx.xx.xx,yy.yy.yy.yy,yy.yy.yy.yy,yy.yy.yy.yy,yy.yy.yy.yy" train.py

-  **参数服务器训练 - 在PaddleCloud上提交任务**

由于厂内Paddlecloud对参数服务器训练做了比较完备的封装，在启动任务时根据配置的参数自动启动服务节点和训练节点。
对于MPI任务，可以通过 \ ``--mpi-nodes``\ 指定服务节点和训练节点的个数；
对于K8S任务，可以通过 \ ``--k8s-cpu-cores``\  和 \ ``—-k8s-ps-cores``\ 指定服务节点和训练节点的个数。启动命令\ ``—-start-cmd``\ 中可以直接使用：

.. code:: sh

   python train.py

.. _fleetrun命令参数介绍:

fleetrun命令参数介绍
----------------

-  GPU模式相关参数:

   -  ips （str，可选）：
      指定选择哪些节点IP进行训练，默认为『127.0.0.1』,
      即会在本地执行单机单卡或多卡训练。
   -  gpus（str, 可选）：
      指定选择哪些GPU卡进行训练，默认为None，即会选择\ ``CUDA_VISIBLE_DEVICES``\ 所显示的所有卡。

-  参数服务器模式可配参数:

   -  server_num（int，可选）：本地模拟分布式任务中，指定参数服务器服务节点的个数
   -  worker_num（int，可选）：本地模拟分布式任务中，指定参数服务器训练节点的个数
   -  servers（str, 可选）：
      多机分布式任务中，指定参数服务器服务节点的IP和端口
   -  workers（str, 可选）：
      多机分布式任务中，指定参数服务器训练节点的IP和端口

-  其他：

   -  log_dir（str, 可选）：
      指定分布式任务训练日志的保存路径，默认保存在“./log/”目录。

.. _使用fleetrun进行GPU多卡训练实例:

使用fleetrun进行GPU多卡训练实例
--------------------------------------------

下面我们将通过例子，为您详细介绍如何利用\ ``fleetrun``\ 将单机单卡训练任务转换为单机多卡训练任务。
这里使用与\ `静态图分布式训练快速开始 <fleet_static_quick_start_cn.rst>`` 相同的示例代码进行说明。
.. code:: py

    import os
    import time
    import paddle
    import paddle.distributed.fleet as fleet
    import paddle.static.nn as nn
    import paddle.fluid as fluid

    def mnist_on_mlp_model():
        train_dataset = paddle.vision.datasets.MNIST(mode='train')
        test_dataset = paddle.vision.datasets.MNIST(mode='test')
        x = paddle.data(name="x", shape=[64, 1, 28, 28], dtype='float32')
        y = paddle.data(name="y", shape=[64, 1], dtype='int64')
        x_flatten = fluid.layers.reshape(x, [64, 784])
        fc_1 = nn.fc(input=x_flatten, size=128, act='tanh')
        fc_2 = nn.fc(input=fc_1, size=128, act='tanh')
        prediction = nn.fc(input=[fc_2], size=10, act='softmax')
        cost = fluid.layers.cross_entropy(input=prediction, label=y)
        acc_top1 = fluid.layers.accuracy(input=prediction, label=y, k=1)
        avg_cost = fluid.layers.mean(x=cost)
        return train_dataset, test_dataset, x, y, avg_cost, acc_top1

    train_data, test_data, x, y, cost, acc = mnist_on_mlp_model()
    place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    train_dataloader = paddle.io.DataLoader(
        train_data, feed_list=[x, y], drop_last=True,
        places=place, batch_size=64, shuffle=True)
    fleet.init(is_collective=True)
    strategy = fleet.DistributedStrategy()
    #optimizer = paddle.optimizer.Adam(learning_rate=0.01)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
    optimizer.minimize(cost)

    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    epoch = 10
    for i in range(epoch):
        total_time = 0
        step = 0
        for data in train_dataloader():
            step += 1
            start_time = time.time()
            loss_val, acc_val = exe.run(
              paddle.static.default_main_program(),
              feed=data, fetch_list=[cost.name, acc.name])
            if step % 200 == 0:
                end_time = time.time()
                total_time += (end_time - start_time)
                print(
                        "epoch: %d, step:%d, train_loss: %f, total time cost = %f, speed: %f"
                    % (i, step, loss_val[0], total_time,
                       1 / (end_time - start_time) ))
单机单卡训练
^^^^^^^^^^^^

将上述代码保存在\ ``res_app.py``\ 代码中，单机单卡训练十分的简单，只需要：

.. code:: sh

   export CUDA_VISIBLE_DEVICES=0
   python res_app.py

可以看见终端上打印日志信息：

.. code:: sh

  epoch: 0, step:200, train_loss: 0.424425, total time cost = 0.000947, speed: 1055.967774
  epoch: 0, step:400, train_loss: 0.273742, total time cost = 0.001725, speed: 1285.413423
  epoch: 0, step:600, train_loss: 0.472131, total time cost = 0.002467, speed: 1347.784062
  epoch: 0, step:800, train_loss: 0.445613, total time cost = 0.003184, speed: 1394.382979
  epoch: 1, step:200, train_loss: 0.512807, total time cost = 0.000681, speed: 1468.593838
  epoch: 1, step:400, train_loss: 0.571385, total time cost = 0.001344, speed: 1508.199928
  epoch: 1, step:600, train_loss: 0.617232, total time cost = 0.002034, speed: 1449.310297
  epoch: 1, step:800, train_loss: 0.392537, total time cost = 0.002813, speed: 1283.446756
  epoch: 2, step:200, train_loss: 0.288508, total time cost = 0.000796, speed: 1256.155735
  epoch: 2, step:400, train_loss: 0.448433, total time cost = 0.001531, speed: 1360.461888
  epoch: 2, step:600, train_loss: 0.593330, total time cost = 0.002292, speed: 1314.005013
    ...

单机多卡训练
^^^^^^^^^^^^

从单机单卡训练到单机多卡训练不需要改动\ ``res_app.py``\ 代码，只需改一行启动命令：

.. code:: sh

   export CUDA_VISIBLE_DEVICES=0,1,2,3
   fleetrun res_app.py

训练日志可以在终端上查看，也可稍后在./log/目录下查看每个卡的日志。
终端可以看到显示日志如下：

.. code:: sh

   -----------  Configuration Arguments -----------
   gpus: 0,1,2,3
   ips: 127.0.0.1
   log_dir: log
   server_num: None
   servers:
   training_script: fleetx_res.py
   training_script_args: []
   worker_num: None
   workers:
   ------------------------------------------------
   INFO 202X-0X-0X 06:09:36,185 launch_utils.py:425] Local start 4 processes. First process distributed environment info (Only For Debug):
   =======================================================================================
               Distributed Envs              Value
   ---------------------------------------------------------------------------------------
   PADDLE_CURRENT_ENDPOINT                   127.0.0.1:33360
   PADDLE_TRAINERS_NUM                       4
   FLAGS_selected_gpus                       0
   PADDLE_TRAINER_ENDPOINTS                  ... 0.1:11330,127.0.0.1:54803,127.0.0.1:49294
   PADDLE_TRAINER_ID                         0
   =======================================================================================
   epoch: 0, step:200, train_loss: 0.306129, total time cost = 0.001170, speed: 854.759323
   epoch: 0, step:400, train_loss: 0.287594, total time cost = 0.002226, speed: 947.009257
   epoch: 0, step:600, train_loss: 0.179934, total time cost = 0.003201, speed: 1025.752996
   epoch: 0, step:800, train_loss: 0.137214, total time cost = 0.005004, speed: 554.582044
   epoch: 1, step:200, train_loss: 0.302534, total time cost = 0.000975, speed: 1025.752996
   epoch: 1, step:400, train_loss: 0.375780, total time cost = 0.001934, speed: 1042.581158
   epoch: 1, step:600, train_loss: 0.247651, total time cost = 0.002892, speed: 1043.878547
   epoch: 1, step:800, train_loss: 0.086278, total time cost = 0.003845, speed: 1049.363022
   .....
