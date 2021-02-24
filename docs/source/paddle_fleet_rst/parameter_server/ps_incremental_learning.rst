增量训练
=====================

简介
---------------------

增量训练是一种常见的机器学习方法，在深度学习领域内有广泛的应用，它代表的是一种动态学习的训练方式，即训练数据随着时间的推移源源不断的加入到当前训练流程中，扩展当前模型的知识和能力。

飞桨的参数服务器训练支持增量训练，支持训练在某一时间点进行训练模型参数(含部分优化器的状态)的全量保存，在重启训练时将之前保存的全量参数进行加载，结合新的训练数据继续训练，从而学习到新的有用信息。


原理介绍
---------------------

飞桨参数服务器增量训练包含两部分内容，即模型保存和模型加载。训练节点分为PServer和Worker两种，每个Worker上都有完整的稠密参数，没有稀疏参数。稀疏参数和稠密参数分布于全部的PServer节点上。


飞桨模型参数在参数服务器下分为稠密参数和稀疏参数两种， 在调用模型保存的接口后，会分别在PServer端和0号Worker端进行参数的保存，其中0号Worker端将保存全部的稠密参数及相关的状态，每个PServer将以分片的形式保存位于该PServer端的稀疏参数。 


飞桨模型参数在参数服务器下分为稠密参数和稀疏参数两种， 需要分别在PServer端和0号Worker端进行加载才能完成对参数的加载。 

训练启动时每个PServer的基本初始流程如下：

- 每个节点执行 `fleet.init_server(dirname=None, var_names=None, **kwargs)` 进行PServer端初始化，分配到此节点的稠密参数会按照定义的形状和初始化方法进行初始化， 稀疏参数则只预定义出初始化方法，稀疏参数会在训练过程中根据前向通信算子发送过来的ID进行实时初始化。 init_server用有两个选配参数，分别是 `dirname`和`var_names`,`dirname`表示需要增量加载的模型路径，两个选配参数相互配合实现稀疏参数的加载，注意, 如果只指定 `dirname`， 则表示会从指定的目录中加载全部的稀疏参数， 如果还指定了`var_names`，则表示加载指定参数名的稀疏参数。 注意，`init_server` 只会加载稀疏参数，稠密参数的加载在Worker端进行。
- 每个节点执行 `fleet.run_server()` 表明当前节点已经初始化成功，可以支持Worker端的连接和通信。


训练启动时每个Worker的基本初始流程如下：

- 每个节点执行 `exe.run(paddle.static.default_startup_program())` 进行参数初始化。
- 0号节点执行 `paddle.static.load_vars()` 指定要加载的稠密参数的名字列表和模型目录，将稠密参数通过此方式进行加载。
- 每个节点执行 `fleet.init_worker()` ， 其中0号节点的稠密参数将同步给相应的PServer，其他节点(非0号)会从PServer端将稠密参数取回本地赋值给本地的稠密参数。


至此，完成了整个训练开始前，PServer和Worker中稠密参数和稀疏参数的加载和同步。



功能效果
---------------------
- 训练开始时，使用上述方法，可实现模型参数的全量加载。
- 训练结束是，使用上述方法，可实现模型参数的全量保存。


使用方法
---------------------

模型保存：

.. code-block:: python

    # 在需要保存模型的地方，执行下面的命令，即可完成模型中全量参数的保存
    # 其中， 稠密参数会被保存在0号Worker上， 稀疏参数会被保存在每个PServer上的同名路径下
    
    dirname = "/you/path/to/model"
    
    if fleet.is_first_worker():
        fleet.save_persistables(dirname)


模型加载(文章末尾附录了获取稀疏/稠密参数的代码，参考或复制使用)：

**对于流式训练，模型加载需要使用全量保存的模型(fleet.save_persistable, 配置mode=0)，如果使用增量保存(Base+Detla)的方式，在拥有准入配置的情况下，可能会丢一部分未被准入的特征。**

.. code-block:: python

    # 模型加载需要区分是PServer还是Worker
    dirname = "/you/path/to/model"
    
    if fleet.is_server():
        sparse_varnames = [var.name for var in get_sparse_vars()]
        fleet.init_server(dirname, sparse_varnames)
        fleet.run_server()

    if fleet.is_worker():
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
    
        exe.run(paddle.static.default_startup_program())
        dense_vars = get_dense_vars()
        paddle.static.load_vars(executor=exe, dirname=path, vars=dense_vars)
        fleet.init_worker()


运行成功提示
---------------------

1. 模型加载当前并没有提示
2. 模型保存成功，会在相应的目录保存下模型文件， 稀疏参数会被保存在每个PServer上的同名路径下。


常见问题与注意事项
---------------------

- 节点动态调整
 + 训练节点在发生变化的情况下， 稀疏参数需要做一次重新分布分布以满足新的加载需求。
 + 当前框架并没有提供此稀疏参数重分布脚本，目前需要用户自行编写。

- 加载指定稠密参数
 + 用户可以选择性的加载所需的稠密参数，具体是在 0号 Worker 执行 `paddle.static.load_vars`时 ，指定的 vars的列表来控制。

- 加载指定稀疏参数
 + 用户可以选择性的加载指定的稀疏参数，具体是在PServer执行`init_server`时，指定`var_names`的列表，通过此列表来控制加载的参数名单。


论文/引用
---------------------
[略]

附录
------------------

获取稀疏/稠密参数的代码
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

        def get_sparse_vars():
            import paddle
            program = paddle.static.default_main_program()
            SPARSE_OP_TYPE_DICT = {"lookup_table", "lookup_table_v2"}

            def is_sparse_op(op):
                if op.type in SPARSE_OP_TYPE_DICT and op.attr('is_sparse') is True or \
                    op.type == "distributed_lookup_table":
                    return True
                return False

            def get_sparse_varnames():
                tablenames = set()
                for op in program.global_block().ops:
                    if is_sparse_op(op):
                        tablenames.add(op.input("W")[0])
                return list(tablenames)

            varnames = get_sparse_varnames()

            sparse_vars = set()
            for varname in varnames:
                sparse_vars.add(program.global_block().vars[varname])
            return list(sparse_vars)

        def get_dense_vars():
            import paddle
            program = paddle.static.default_main_program()

            def is_persistable(var):
                if var.desc.type() == paddle.fluid.core.VarDesc.VarType.FEED_MINIBATCH or \
                   var.desc.type() == paddle.fluid.core.VarDesc.VarType.FETCH_LIST or \
                   var.desc.type() == paddle.fluid.core.VarDesc.VarType.READER:
                    return False
                return var.persistable

            exe = paddle.static.Executor(paddle.CPUPlace())
            sparse_varnames = [var.name for var in get_sparse_vars()]
            dense_vars = set()
            for name, var in program.global_block().vars.items():
                if is_persistable(var) and var.name not in sparse_varnames:
                    dense_vars.add(var)
            return list(dense_vars)

