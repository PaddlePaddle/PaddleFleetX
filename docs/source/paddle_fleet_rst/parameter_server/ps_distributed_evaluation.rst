分布式预测
==================

简介
------
分布式预测任务将预测数据均匀分布式在多台机器上，每台机器仅预测整个数据集的一部分，节点之间通过 `all_reduce` 等集合通信操作完成各自预测结果的同步，从而获取整个数据集的预测结果。

为什么要做分布式预测，除了通过数据并行的方式节省预测时间外，另一个很重要的原因是，在某些场景，例如推荐系统或者搜索引擎中， 稀疏参数（embedding）的 *feature id* 可能会非常多，当 *feature id* 达到一定数量时，稀疏参数会变得很大以至于单机内存无法存放，从而导致无法预测。

原理
------
分布式预测的原理基本和分布式训练一致，都将节点分为 **Worker** 和 **PServer** 两类，这两类节点在训练任务和预测任务中的分工如下：

    - **Worker**\ ：在训练时，Worker负责完成训练数据读取、从PServer上拉取稀疏参数然后进行前向网络计算、反向梯度计算等过程，并将计算出的梯度上传至PServer。在预测时，Worker负责完成预测数据读取、从PServer上拉取稀疏参数然后进行前向计算。所有Worker间可进行集合通信，从而获取全局的预测结果。
    - **PServer**\ ：在训练时，PServer在收到训练Worker传来的梯度后，会根据指定的优化器完成更新参数，并将参数发送给训练Worker。在预测时，PServer仅作为稀疏参数存储器，响应预测Worker拉取稀疏参数的请求。

分布式预测任务的流程主要有以下三步：
   
    1. 自定义预测组网
    2. 初始化分布式集群环境，加载模型参数。
    3. 生成分布式预测组网，自定义reader，开始预测。

分布式预测功能主要通过 `DistributedInfer` 工具类完成，下面对相关API的功能和参数进行介绍。

.. py:class:: paddle.distributed.fleet.utils.ps_util.DistributedInfer(main_program=None, startup_program=None)

    PaddlePaddle的分布式预测工具类。

    **参数：**
        - main_program(paddle.static.Program, optional)，单机预测组网，若为None，则认为 `paddle.static.default_main_program()` 为单机预测组网。默认为None。
        - startup_program(paddle.static.Program, optional)，单机预测初始化组网，若为None，则认为 `paddle.static.default_startup_program()` 为单机预测初始化组网。默认为None。

    **方法：**

    .. py:method:: init_distributed_infer_env(exe, loss, role_maker=None, dirname=None)

        初始化分布式集群环境，加载模型参数。需要注意，该接口仅在纯分布式预测的任务中才需要被调用，在先训练后预测的分布式一体任务里，此接口无需调用，且不会生效。

        **参数：**
            - exe, (paddle.static.Executor, required)，初始化分布式集群环境时需要用到的网络执行器。
            - loss, (Tensor, required)， 预测网络 `loss` 变量。
            - role_maker, (RoleMakerBase, optional)， 分布式训练（预测）任务环境配置，若为None，则框架会自动根据用户在环境变量中的配置进行分布式训练（预测）环境的初始化。默认为None。
            - dirname, (String, optional)， 参数路径。若为None，则不加载参数。默认为None。

    .. py:method:: get_dist_infer_program():

        生成分布式预测组网。相较于单机预测组网，两者区别仅在于：将稀疏参数查询操作替换为分布式稀疏参数查询操作，即将 `lookup_table` 算子替换为 `distributed_lookup_table` 。

        **返回：**
            Program，分布式预测组网。

使用方法
--------

分布式预测常见的应用场景有以下两种，分布式训练+预测一体任务，及独立的分布式预测任务，两种任务的特点分别为：
    
    - **分布式训练 + 预测一体任务**：指分布式训练结束后，Worker节点不向PServer发送任务结束通知，而是继续开始预测。这类任务在进行预测时，分布式集群环境已经初始化好，且不需要进行参数加载。
    - **分布式预测任务**：指纯预测的分布式任务。这类任务在进行预测时，分布式集群环境还未初始化好，且往往需要进行参数加载。

下面分别介绍对这两种分布式预测任务的使用方法：

分布式训练 + 预测一体任务
~~~~~~~~~~~~~~~~~~~~~~~~~
 
.. code:: python

    ...
    model = WideDeepModel()
    model.net(is_train=True)

    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()
    else:
        exe.run(paddle.default_startup_program())
        fleet.init_worker()

        # 分布式训练
        distributed_training(exe, model)

        # 1. 生成单机预测组网
        test_main_program = paddle.static.Program()
        test_startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program=test_main_program, startup_program=test_startup_program):
            with paddle.utils.unique_name.guard():
                model.net(is_train=False)
        
        # 2. 生成分布式预测组网，定义reader，进行预测
        dist_infer = DistributedInfer(main_program=test_main_program, startup_program=test_startup_program)
        dist_infer_program = dist_infer.get_dist_infer_program()
        
        test_data = WideDeepDataset(data_path="./data")
        reader = model.loader.set_sample_generator(test_data, batch_size=batch_size, drop_last=True, places=place)
        
        reader.start()
        batch_idx = 0
        try:
            while True:
                loss_val = exe.run(program=dist_infer_program,
                                    fetch_list=[model.cost.name])
                if batch_idx % 10 == 0:
                    loss_val = np.mean(loss_val)
                    message = "TEST ---> batch_idx: {} loss: {}\n".format(batch_idx, loss_val)  
        except fluid.core.EOFException:
            reader.reset()

        fleet.stop_worker()

分布式预测任务
~~~~~~~~~~~~~~~~~

.. code:: python

    ...

    # 1. 定义单机预测组网
    model = WideDeepModel()
    model.net(is_train=False)

    # 2. 初始化分布式预测环境，加载模型参数
    dist_infer = DistributedInfer(main_program=test_main_program, startup_program=test_startup_program)
    exe = paddle.static.Executor()
    dirname = "./init_params/"
    dist_infer.init_distributed_infer_env(exe, model.cost, dirname=dirname)
   
    # 3.生成分布式预测组网，定义reader，进行预测
    if fleet.is_worker():
        dist_infer_program = dist_infer.get_dist_infer_program()
        
        test_data = WideDeepDataset(data_path="./data")
        reader = model.loader.set_sample_generator(test_data, batch_size=batch_size, drop_last=True, places=place)
        
        reader.start()
        batch_idx = 0
        try:
            while True:
                loss_val = exe.run(program=dist_infer_program,
                                    fetch_list=[model.cost.name])
                if batch_idx % 10 == 0:
                    loss_val = np.mean(loss_val)
                    message = "TEST ---> batch_idx: {} loss: {}\n".format(batch_idx, loss_val)
                    print(message)
        except fluid.core.EOFException:
            reader.reset()
        
        fleet.stop_worker()

运行方法
~~~~~~~~~~~~

完整运行示例见 `examples/wide_and_deep`。该示例为分布式训练 + 预测一体任务。

配置完成后，通过\ ``fleetrun``\ 指令运行分布式任务。命令示例如下，其中\ ``server_num``, ``worker_num``\ 分别为服务节点和训练节点的数量。

.. code:: sh

   fleetrun --server_num=2 --worker_num=2 train.py
