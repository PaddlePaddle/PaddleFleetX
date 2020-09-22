动态图分布式训练快速开始
------------------------

`Paddle官方文档 <https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/tutorial/quick_start/dynamic_graph/dynamic_graph.html>`__\ 中对动态图（命令式编程）做了比较详细的介绍。Paddle的分布式高级API \ ``paddle.distributed.fleet``\ 接口从Paddle
2.0-RC版本开始支持动态图分布式任务执行。本篇文章我们将介绍如何使用 \ ``paddle.distributed.fleet``\ 接口进行动态图分布式训练。接下来我们以一个简单全连接网络实例为例，说明如何将单机单卡训练改成分布式单机多卡训练，再到多机多卡训练。

注：目前\ ``paddle.distributed.fleet``\ 启动动态图分布式训练仅支持集合通信（Colletive）模式，不支持参数服务器（Parameter-Server）模式。本文示例为集合通信（Colletive）模式任务。

版本要求
~~~~~~~~

-  paddlepaddle 2.0-rc-gpu版本及以上

单机单卡训练
~~~~~~~~~~~~

下面是一个非常简单的动态图单机单卡程序。网络只有只有2层全连接层，用均方差误差（MSELoss）作为损失函数，Adam优化器进行参数的更新。循环迭代20轮中，每轮打印出当前网络具体的损失值。

.. code:: py

   import paddle
   import paddle.nn as nn

   # 定义全连接网络，需继承自nn.Layer
   class LinearNet(nn.Layer):
       def __init__(self):
           super(LinearNet, self).__init__()
           self._linear1 = nn.Linear(10, 10)
           self._linear2 = nn.Linear(10, 1)

       def forward(self, x):
           return self._linear2(self._linear1(x))


   # 1.开启动态图模式
   paddle.disable_static()

   # 2. 定义网络对象，损失函数和优化器
   layer = LinearNet()
   loss_fn = nn.MSELoss()
   adam = paddle.optimizer.Adam(
   learning_rate=0.001, parameters=layer.parameters())


   for step in range(20):
       # 3. 执行前向网络
       inputs = paddle.randn([10, 10], 'float32')
       outputs = layer(inputs)
       labels = paddle.randn([10, 1], 'float32')
       loss = loss_fn(outputs, labels)

       print("step:{}\tloss:{}".format(step, loss.numpy()))

       # 4. 执行反向计算和参数更新
       loss.backward()
       adam.step()
       adam.clear_grad()

将以上代码保存为\ ``train_single.py``\ ，运行\ ``python train_single.py``\ ，您将看到显示如下日志信息：

::

   step:0  loss:[1.2709768]
   step:1  loss:[0.7705929]
   step:2  loss:[2.2044802]
   step:3  loss:[1.6021421]
   step:4  loss:[2.0286825]
   step:5  loss:[0.7866151]
   step:6  loss:[1.926115]
   step:7  loss:[0.3647427]
   ...

.. _单机单卡训练-1:

单机多卡训练
~~~~~~~~~~~~

使用Fleet接口进行动态图分布式训练其实非常的简单，只需修改4个步骤： 1.
导入\ ``paddle.distributed.fleet``\ 包

.. code:: py

   from paddle.distributed import fleet

2. 初始化fleet环境

.. code:: py

   fleet.init(is_collective=True)

3. 通过fleet获取分布式优化器和分布式模型

.. code:: py

   adam = fleet.distributed_optimizer(adam)
   dp_layer = fleet.distributed_model(layer)

4. 在执行反向（backward函数）前后进行损失缩放和反向梯度的聚合

.. code:: py

   loss = dp_layer.scale_loss(loss)
   loss.backward()
   dp_layer.apply_collective_grads()

根据我们最开始提供的单机单卡代码示例，再根据4步口诀进行修改，完整的单机多卡示例代码如下：

.. code:: py

   import paddle
   import paddle.nn as nn
   #分布式step 1: 导入paddle.distributed.fleet包
   from paddle.distributed import fleet

   # 定义全连接网络，需继承自nn.Layer
   class LinearNet(nn.Layer):
       def __init__(self):
           super(LinearNet, self).__init__()
           self._linear1 = nn.Linear(10, 10)
           self._linear2 = nn.Linear(10, 1)

       def forward(self, x):
           return self._linear2(self._linear1(x))


   # 1.开启动态图模式
   paddle.disable_static()

   # 分布式step 2: 初始化fleet
   fleet.init(is_collective=True)

   # 2. 定义网络对象，损失函数和优化器
   layer = LinearNet()
   loss_fn = nn.MSELoss()
   adam = paddle.optimizer.Adam(
   learning_rate=0.001, parameters=layer.parameters())

   # 分布式step 3: 通过fleet获取分布式优化器和分布式模型
   adam = fleet.distributed_optimizer(adam)
   dp_layer = fleet.distributed_model(layer)


   for step in range(20):
       # 3. 执行前向网络
       inputs = paddle.randn([10, 10], 'float32')
       outputs = dp_layer(inputs)
       labels = paddle.randn([10, 1], 'float32')
       loss = loss_fn(outputs, labels)

       print("step:{}\tloss:{}".format(step, loss.numpy()))

       # 4. 执行反向计算和参数更新
       # 分布式step 4: 在执行反向（backward函数）前后进行损失缩放和反向梯度的聚合
       loss = dp_layer.scale_loss(loss)
       loss.backward()
       dp_layer.apply_collective_grads()

       adam.step()
       adam.clear_grad()

将以上代码保存为\ ``train_fleet.py``\ ，假设要运行2卡的任务，那么只需在命令行中执行:

.. code:: sh

   fleetrun --gpus=0,1 dygraph_fleet.py

您将看到显示如下日志信息：

::

   -----------  Configuration Arguments -----------
   gpus: 0,1
   ips: 127.0.0.1
   log_dir: log
   server_num: None
   servers:
   training_script: dygraph_fleet.py
   training_script_args: []
   worker_num: None
   workers:
   ------------------------------------------------
   INFO 2020-0X-XX 08:33:30,247 launch.py:441] Run collective gpu mode. gpu arguments:['--gpus'], cuda count:8
   INFO 2020-0X-XX 08:33:30,247 launch_utils.py:430] Local start 2 processes. First process distributed environment info (Only For Debug):
      +=======================================================================================+
      |                        Distributed Envs                      Value                    |
      +---------------------------------------------------------------------------------------+
      |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:59664               |
      |                     PADDLE_TRAINERS_NUM                        2                      |
      |                     FLAGS_selected_gpus                        0                      |
      |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:59664,127.0.0.1:48993       |
      |                       PADDLE_TRAINER_ID                        0                      |
      +=======================================================================================+
   step:0  loss:[1.3279431]
   step:1  loss:[2.5023699]
   step:2  loss:[3.3197324]
   step:3  loss:[2.6869867]
   step:4  loss:[2.6306524]
   step:5  loss:[1.9267073]
   step:6  loss:[1.2037501]
   step:7  loss:[1.1434236]
   ...

完整2卡的日志信息也可在\ ``./log/``\ 目录下查看。了解更多\ ``fleetrun``\ 的用法可参考左侧文档\ ``fleetrun 启动分布式任务``\ 。

多机多卡训练
~~~~~~~~~~~~

从单机多卡到多机多卡训练，在代码上并不需要做任何改动，只需修改启动命令，以2机4卡为例：

.. code:: sh

   fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus=0,1 dygraph_fleet.py

在2台机器上\ **分别**\ 运行以上启动命令，\ ``fleetrun``\ 将在后台分别启动2个多进程任务，执行分布式多机训练。
您将在ip为xx.xx.xx.xx的机器上看到命令台输出日志信息：

::

   -----------  Configuration Arguments -----------
   gpus: None
   ips: xx.xx.xx.xx,yy.yy.yy.yy
   log_dir: log
   server_num: None
   servers:
   training_script: dygraph_fleet.py
   training_script_args: []
   worker_num: None
   workers:
   ------------------------------------------------
   INFO 2020-0X-XX 21:29:41,918 launch.py:434] Run collective gpu mode. gpu arguments:['--ips'], cuda count:2
   INFO 2020-0X-XX 21:29:41,919 launch_utils.py:426] Local start 2 processes. First process distributed environment info (Only For Debug):
       +=======================================================================================+
       |                        Distributed Envs                      Value                    |
       +---------------------------------------------------------------------------------------+
       |                 PADDLE_CURRENT_ENDPOINT               xx.xx.xx.xx:6070              |
       |                     PADDLE_TRAINERS_NUM                        4                      |
       |                     FLAGS_selected_gpus                        0                      |
       |                PADDLE_TRAINER_ENDPOINTS  ... :6071,yy.yy.yy.yy:6070,yy.yy.yy.yy:6071|
       |                       PADDLE_TRAINER_ID                        0                      |
       +=======================================================================================+
   step:0  loss:[5.2519045]
   step:1  loss:[3.139771]
   step:2  loss:[2.0075738]
   step:3  loss:[1.4674551]
   step:4  loss:[4.0751777]
   step:5  loss:[2.6568782]
   step:6  loss:[1.1998112]
   ...

同样完整的日志信息也分别在xx.xx.xx.xx机器和yy.yy.yy.yy机器上的\ ``./log/``\ 目录下查看。

小结
~~~~

至此，相信您已经通过4步口诀掌握了如何将一个普通的paddle动态图单卡任务转换为多卡任务。推荐使用单卡进行调试，真正执行训练时切换为多卡任务。我们也将在未来继续完善Fleet动态图模块，通过与静态图类似的方式实现分布式训练任务在不同场景下的优化，敬请期待！
