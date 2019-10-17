
使用FleetAPI进行分布式训练
==========================

FleetAPI 设计说明
-----------------

Fleet是PaddlePaddle分布式训练的高级API。Fleet的命名出自于PaddlePaddle，象征一个舰队中的多只双桨船协同工作。Fleet的设计在易用性和算法可扩展性方面做出了权衡。用户可以很容易从单机版的训练程序，通过添加几行代码切换到分布式训练程序。此外，分布式训练的算法也可以通过Fleet
API接口灵活定义。具体的设计原理可以参考\ `Fleet
API设计文档 <https://github.com/PaddlePaddle/Fleet/blob/develop/README.md>`_\ 。当前FleetAPI还处于paddle.fluid.incubate目录下，未来功能完备后会放到paddle.fluid目录中，欢迎持续关注。

Fleet API快速上手示例
---------------------

下面会针对Fleet
API最常见的两种使用场景，用一个模型做示例，目的是让用户有快速上手体验的模板。快速上手的示例源代码可以在\ `Fleet Quick Start <https://github.com/PaddlePaddle/Fleet/tree/develop/examples/quick-start>`_ 找到。


* 
  假设我们定义MLP网络如下：

  .. code-block:: python

     import paddle.fluid as fluid

     def mlp(input_x, input_y, hid_dim=128, label_dim=2):
       fc_1 = fluid.layers.fc(input=input_x, size=hid_dim, act='tanh')
       fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim, act='tanh')
       prediction = fluid.layers.fc(input=[fc_2], size=label_dim, act='softmax')
       cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
       avg_cost = fluid.layers.mean(x=cost)
       return avg_cost

* 
  定义一个在内存生成数据的Reader如下：

  .. code-block:: python

     import numpy as np

     def gen_data():
         return {"x": np.random.random(size=(128, 32)).astype('float32'),
                 "y": np.random.randint(2, size=(128, 1)).astype('int64')}

* 
  单机Trainer定义

  .. code-block:: python

     import paddle.fluid as fluid
     from nets import mlp
     from utils import gen_data

     input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
     input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

     cost = mlp(input_x, input_y)
     optimizer = fluid.optimizer.SGD(learning_rate=0.01)
     optimizer.minimize(cost)
     place = fluid.CUDAPlace(0)

     exe = fluid.Executor(place)
     exe.run(fluid.default_startup_program())
     step = 1001
     for i in range(step):
       cost_val = exe.run(feed=gen_data(), fetch_list=[cost.name])
       print("step%d cost=%f" % (i, cost_val[0]))

* 
  Parameter Server训练方法

  参数服务器方法对于大规模数据，简单模型的并行训练非常适用，我们基于单机模型的定义给出使用Parameter Server进行训练的示例如下：

  .. code-block:: python

     import paddle.fluid as fluid
     from nets import mlp
     from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
     from paddle.fluid.incubate.fleet.base import role_maker
     from utils import gen_data

     input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
     input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

     cost = mlp(input_x, input_y)
     optimizer = fluid.optimizer.SGD(learning_rate=0.01)

     role = role_maker.PaddleCloudRoleMaker()
     fleet.init(role)
     optimizer = fleet.distributed_optimizer(optimizer)
     optimizer.minimize(cost)

     if fleet.is_server():
       fleet.init_server()
       fleet.run_server()
     elif fleet.is_worker():
       place = fluid.CPUPlace()
       exe = fluid.Executor(place)
       exe.run(fluid.default_startup_program())
       step = 1001
       for i in range(step):
         cost_val = exe.run(
             program=fluid.default_main_program(),
             feed=gen_data(),
             fetch_list=[cost.name])
         print("worker_index: %d, step%d cost = %f" %
              (fleet.worker_index(), i, cost_val[0]))

* 
  Collective训练方法

  Collective Training通常在GPU多机多卡训练中使用，一般在复杂模型的训练中比较常见，我们基于上面的单机模型定义给出使用Collective方法进行分布式训练的示例如下：

  .. code-block:: python

     import paddle.fluid as fluid
     from nets import mlp
     from paddle.fluid.incubate.fleet.collective import fleet
     from paddle.fluid.incubate.fleet.base import role_maker
     from utils import gen_data

     input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
     input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

     cost = mlp(input_x, input_y)
     optimizer = fluid.optimizer.SGD(learning_rate=0.01)
     role = role_maker.PaddleCloudRoleMaker(is_collective=True)
     fleet.init(role)

     optimizer = fleet.distributed_optimizer(optimizer)
     optimizer.minimize(cost)
     place = fluid.CUDAPlace(0)

     exe = fluid.Executor(place)
     exe.run(fluid.default_startup_program())
     step = 1001
     for i in range(step):
       cost_val = exe.run(
           program=fluid.default_main_program(),
           feed=gen_data(),
           fetch_list=[cost.name])
       print("worker_index: %d, step%d cost = %f" %
            (fleet.worker_index(), i, cost_val[0]))

更多使用示例
------------

`点击率预估 <https://github.com/PaddlePaddle/Fleet/tree/develop/examples/ctr>`_

`语义匹配 <https://github.com/PaddlePaddle/Fleet/tree/develop/examples/semantic_matching>`_

`向量学习 <https://github.com/PaddlePaddle/Fleet/tree/develop/examples/word2vec>`_

`基于Resnet50的图像分类 <https://github.com/PaddlePaddle/Fleet/tree/develop/examples/resnet50>`_

`基于Transformer的机器翻译 <https://github.com/PaddlePaddle/Fleet/tree/develop/examples/transformer>`_

`基于Bert的语义表示学习 <https://github.com/PaddlePaddle/Fleet/tree/develop/examples/bert>`_

Fleet API相关的接口说明
-----------------------

Fleet API接口
^^^^^^^^^^^^^


* init(role_maker=None)

  * fleet初始化，需要在使用fleet其他接口前先调用，用于定义多机的环境配置

* is_worker()

  * Parameter Server训练中使用，判断当前节点是否是Worker节点，是则返回True，否则返回False

* is_server(model_dir=None)

  * Parameter Server训练中使用，判断当前节点是否是Server节点，是则返回True，否则返回False

* init_server()

  * Parameter Server训练中，fleet加载model_dir中保存的模型相关参数进行parameter
    server的初始化

* run_server()

  * Parameter Server训练中使用，用来启动server端服务

* init_worker()

  * Parameter Server训练中使用，用来启动worker端服务

* stop_worker()

  * 训练结束后，停止worker

* distributed_optimizer(optimizer, strategy=None)

  * 分布式优化算法装饰器，用户可带入单机optimizer，并配置分布式训练策略，返回一个分布式的optimizer

RoleMaker
^^^^^^^^^


* 
  MPISymetricRoleMaker


  * 
    描述：MPISymetricRoleMaker会假设每个节点启动两个进程，1worker+1pserver，这种RoleMaker要求用户的集群上有mpi环境。

  * 
    示例：

    .. code-block:: python

       from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
       from paddle.fluid.incubate.fleet.base import role_maker

       role = role_maker.MPISymetricRoleMaker()
       fleet.init(role)

  * 
    启动方法：

    .. code-block:: python

       mpirun -np 2 python trainer.py

* 
  PaddleCloudRoleMaker


  * 
    描述：PaddleCloudRoleMaker是一个高级封装，支持使用paddle.distributed.launch或者paddle.distributed.launch_ps启动脚本

  * 
    Parameter Server训练示例：

    .. code-block:: python

       from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
       from paddle.fluid.incubate.fleet.base import role_maker

       role = role_maker.PaddleCloudRoleMaker()
       fleet.init(role)

  * 
    启动方法：

    .. code-block:: python

       python -m paddle.distributed.launch_ps --worker_num 2 --server_num 2 trainer.py

  * 
    Collective训练示例：

    .. code-block:: python

       from paddle.fluid.incubate.fleet.collective import fleet
       from paddle.fluid.incubate.fleet.base import role_maker

       role = role_maker.PaddleCloudRoleMaker(is_collective=True)
       fleet.init(role)

  * 
    启动方法：

    .. code-block:: python

        python -m paddle.distributed.launch trainer.py

* 
  UserDefinedRoleMaker


  * 
    描述：用户自定义节点的角色信息，IP和端口信息

  * 
    示例：

    .. code-block:: python

       from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
       from paddle.fluid.incubate.fleet.base import role_maker

       role = role_maker.UserDefinedRoleMaker(
                   current_id=int(os.getenv("CURRENT_ID")),
                   role=role_maker.Role.WORKER if bool(int(os.getenv("IS_WORKER"))) 
                                                                                   else role_maker.Role.SERVER,
                   worker_num=int(os.getenv("WORKER_NUM")),
                   server_endpoints=pserver_endpoints)
       fleet.init(role)

Strategy
^^^^^^^^


* Parameter Server Training

  * Sync_mode

* Collective Training

  * LocalSGD
  * ReduceGrad

Fleet Mode
^^^^^^^^^^


* 
  Parameter Server Training

  .. code-block:: python

     from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet

* 
  Collective Training

  .. code-block:: python

     from paddle.fluid.incubate.fleet.collective import fleet
