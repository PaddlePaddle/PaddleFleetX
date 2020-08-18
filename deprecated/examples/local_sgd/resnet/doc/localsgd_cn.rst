Adaptive Communication Strategy in Local SGD
=============

背景
---------
GPU多机多卡同步训练过程中存在慢trainer现象，即每步中训练快的trainer的同步通信需要等待训练慢的trainer。由于每步中慢trainer的运行时间具有随机性，每次训练都需要等待最慢的trainer再进行通信同步，从而性能下降，如下图所
示。
因此我们使用局部异步训练的方式——LocalSGD，通过多步异步训练（无通信阻塞）实现慢trainer时间均摊，从而提升同步训练性能。

.. image:: image/localsgd.png


原理
---------
LocalSGD采用多个step之后再同步参数，该方法一方面减少通信次数，提高吞吐量；另一方面通过多步异步训练，实现trainer时间均摊，减少通信同步的等待时间。具体的算法步骤 \ :sup:`[1]` 如下所示：

.. image:: image/lsgd_algorithm.png

其中同步步长（ :math:`K` ）参数设置是人为设定的，该参数影响了整个模型的精度和速度。可以看出 :math:`K` 越大，通信开销减少，但随着同步次数减少，模型精度下降。因此,我们采用自适应步长的方法可以有效避免人为
设置的不确定性，兼顾速度和精度，提高整体性能。该方法的主要思想是在模型参数变化剧烈的时候，减少 :math:`K` ，通过更多的同步通信，而保证模型收敛以及精度；在模型参数趋于稳定的时候，增大 :math:`K` ，从而减少通信次数，提高模型吞吐量。

为了衡量模型参数的变化程度，文献 \ :sup:`[2]` 采用学习率和训练损失，从而得到自适应的训练步长，第l轮的步长 :math:`K_{l}` 的计算公式如下：

.. math::

   K_{l}=\left[\sqrt{\frac{\eta_{0}}{\eta_{l}} \frac{F\left(\mathbf{x}_{t=l}\right)}{F\left(\mathbf{x}_{t=0}\right)}} K_{0}\right]

其中，:math:`\eta_{l}` 和 :math:`\eta_{0}` 分别是当前第l轮的学习率和初始的学习率。 :math:`F\left(\mathbf{x}_{t=l}\right)`  和 :math:`F\left(\mathbf{x}_{t=l}\right)` 分别为
第l轮的训练损失和初始的损失。

自定义步长LocalSGD训练
---------
自定义步长LocalSGD训练方式主要有三个参数，分别是：

..  csv-table::
    :header: "选项", "类型", "可选值", "说明"
    :widths: 3, 3, 3, 5

    ":code:`use_local_sgd`", "bool", "False/True", "是否开启Local SGD，默认不开启"
    ":code:`local_sgd_is_warm_steps`", "int", "大于0", "训练多少轮之后才使用Local SGD方式训练"
    ":code:`local_sgd_steps`", "int", "大于0", "Local SGD的步长"

说明：

- Local SGD的warmup步长 :code:`local_sgd_is_warm_steps` 影响最终模型的泛化能力，一般需要等到模型参数稳定之后在进行Local SGD训练，经验值可以将学习率第一次下降时的epoch作为warmup步长，之后再进行Local SGD训练。
- Local SGD步长 :code:`local_sgd_steps` ，一般该值越大，通信次数越少，训练速度越快，但随之而来的时模型精度下降。经验值设置为2或者4。

通过设置上述三个参数即可实现LocalSGD训练，只需要在原分布式训练代码添加几个部分：

**1、设置分布策略**

在训练策略中，选择打开 :code:`use_local_sgd` 开关。

.. code-block:: python

            dist_strategy = DistributedStrategy()
            # 设置Local SGD模式
            dist_strategy.use_local_sgd = True


**2、定义warmup过程**

需要自定义warmup策略，在前 :code:`local_sgd_warmup` 轮数中，仍然使用普通的分布式SGD训练。

.. code-block:: python

            # 定义warmup过程
            def get_local_sgd_steps(passid, local_sgd_steps, local_sgd_warmup):
                offset = passid - local_sgd_warmup
                if offset < 0:
                    return 1
                warm_up = [2 ** i for i in range(local_sgd_steps) if 2 ** i <=local_sgd_steps]
                if offset >= len(warm_up):
                    return local_sgd_steps
                else:
                    return warm_up[offset]

**3、添加转换Program代码**

这里主要有两个program: 一个是 :code:`fleet._origin_program` ，其没有通信类op，例如 :code:`c_allreduce_sum` 等；
另一个是 :code:`fleet.main_program` ，包含通信类op，执行分布式SGD训练。通过在不同的 :code:`local_sgd_steps`
切换不同的program，从而实现减少通信次数的目的。

.. code-block:: python

            # 获取当前轮的local steps
            cur_local_sgd = get_local_sgd_steps(pass_id, local_sgd_steps, local_sgd_is_warm_steps)
            # 通过step_cnt，切换不同的program
            if step_cnt % cur_local_sgd == 0:
                current_prog = fleet.main_program
            else:
                current_prog = fleet._origin_program
            loss, acc1, acc5, lr = train_exe.run(current_prog, fetch_list=train_fetch_list, use_program_cache=True)


完整的Local SGD的训练代码可以参考：
https://github.com/PaddlePaddle/Fleet/tree/develop/examples/local_sgd/resnet


自适应步长LocalSGD训练方式
---------
自适应步长LocalSGD需要依赖于学习率，因此只适用于SGD等可以获取全局学习率的优化方法，而无法应用于Adam等方法。相较于
自定义步长LocalSGD训练方式而言，该方法不需要设置local step步长参数 :code:`local_sgd_steps` 以及warmup步长参数 :code:`local_sgd_is_warm_steps` 。
相应的，需要添加获取当前的训练损失以及当前的学习率的代码。具体的添加步骤如下：

**1、获取当前的训练损失**

由于是分布式训练每张卡的训练损失不一致，因此需要在每一轮结束的时候，同步各自的训练损失。

.. code-block:: python

            # 组一个实现同步训练损失的网络
            def build_allreduce_program(main_prog, startup_program):
                ring_id = 0
                with fluid.program_guard(main_prog, startup_program):
                    tindata = fluid.layers.data(
                        name="tindata", shape=[1], dtype='float32')
                    toutdata = main_prog.current_block().create_var(
                        name="outofallreduce",
                        dtype='float32',
                        type=core.VarDesc.VarType.LOD_TENSOR,
                        persistable=False,
                        stop_gradient=False)
                    main_prog.global_block().append_op(
                        type="c_allreduce_sum",
                        inputs={'X': tindata},
                        attrs={'ring_id': ring_id},
                        outputs={'Out': toutdata})
                    main_prog.global_block().append_op(
                        type="c_sync_comm_stream",
                        inputs={'X': toutdata},
                        outputs={'Out': toutdata},
                        attrs={'ring_id': ring_id})
                    return toutdata       
            # 初始化
            all_train_prog = fluid.Program()
            all_startup_prog = fluid.Program()
            result = build_allreduce_program(all_train_prog, all_startup_prog)
            all_place = fluid.CUDAPlace(gpu_id)
            all_exe = fluid.Executor(all_place)
            all_exe.run(all_startup_prog)

**2、自适应获取当前步长**

根据当前训练损失、初始损失、当前学习率和初始学习率，计算得到当前的训练步长。

.. code-block:: python

            # 定义自适应获取训练步长
            def adaptive_local_step(ini_loss, ini_lr, cur_loss, cur_lr, base_step, pre_step):
                # 参考文献: https://arxiv.org/pdf/1810.08313.pdf
                inf_loss = 0.6
                fir = ini_lr * (cur_loss - inf_loss)
                sec = cur_lr * max((ini_loss - inf_loss), 1e-12)
                ratio = fir / sec
                step = int(base_step * math.sqrt(ratio))
                if step < 1:
                    step = 1
                if step > pre_step + 20:
                    step = pre_step
                return step


**3、添加转换Program代码**

与自定义步长Local SGD类似。

.. code-block:: python

            # 获取当前轮的训练损失
            all_loss = all_exe.run(all_train_prog,
                      feed={'tindata': train_loss},
                      fetch_list=[result.name])
            reduce_loss = float(all_loss[0]) / num_trainers
            # 获取当前的local step
            cur_local_sgd = adaptive_local_step(ini_loss, ini_lr, cur_loss, cur_lr, base_step, pre_step)
            # 保存前一轮的step，防止训练波动，导致local step变化剧烈
            pre_step = cur_local_sgd
            # 通过step_cnt，切换不同的program
            if step_cnt % cur_local_sgd == 0:
                current_prog = fleet.main_program
            else:
                current_prog = fleet._origin_program
            loss, acc1, acc5, lr = train_exe.run(current_prog, fetch_list=train_fetch_list, use_program_cache=True)

完整的训练方法可以参考：
https://github.com/PaddlePaddle/Fleet/tree/develop/examples/local_sgd/resnet

参考文献
---------
[1] Lin T, Stich S U, Patel K K, et al. Don't Use Large Mini-Batches, Use Local SGD[J]. arXiv preprint arXiv:1808.07217, 2018.

[2] Wang J, Joshi G. Adaptive communication strategies to achieve the best error-runtime trade-off in local-update SGD[J]. arXiv preprint arXiv:1810.08313, 2018.

