Adaptive Communication Strategy in LocalSGD
=============

Background
---------

There is a slow trainer phenomenon in the GPU multi-machine multi-card synchronous training process, 
that is, the synchronous communication of the fast-training trainer in each step needs to wait for 
the slow-training trainer. Due to the randomness of the running time of the slow trainer in each step, 
each training needs to wait for the slowest trainer before performing communication synchronization, 
thereby reducing performance, as shown below.

Therefore, we use the local asynchronous training method, **LocalSGD**, to achieve the equalization 
of slow trainer time through multi-step initial training (two-way without communication), thereby 
improving synchronous training performance.

.. image:: image/localsgd.png


Theory
---------

LocalSGD uses multiple steps to synchronize parameters. 
This method reduces communication times and improves throughput. 
On the other hand, through multi-step asynchronous training, 
the trainer time is evenly distributed and the waiting time for 
communication synchronization is reduced. 
The specific algorithm steps \ :sup:`[1]`  are as follows:


.. image:: image/lsgd_algorithm.png


The synchronization step size ( :math:`K` ) parameter setting is artificial setting, 
and this parameter affects the accuracy and speed of the entire model. 
It can be clearly seen that the larger :math:`K` , the lower the communication overhead, 
but as the number of synchronizations decreases, the accuracy of the model decreases significantly. 
Therefore, we adopt the adaptive step size method to effectively avoid artificial setting uncertainty. 
The main principle of adaptive step communication is to reduce :math:`K` when the model 
parameters change drastically, and to ensure the model convergence and accuracy 
through more synchronization parameters; when the model parameters are stable, 
increase :math:`K` to reduce the number of communications and improve model throughput.
In order to measure the degree of change of the model parameters, 
the reference  \ :sup:`[2]` uses the learning rate and training loss to 
obtain the adaptive training step size. The calculation formula of the step size of 
the l round :math:`K_{l}` is as follows:


.. math::

   K_{l}=\left[\sqrt{\frac{\eta_{0}}{\eta_{l}} \frac{F\left(\mathbf{x}_{t=l}\right)}{F\left(\mathbf{x}_{t=0}\right)}} K_{0}\right]

Among them, :math:`\ eta_ {l}` and :math:`\ eta_ {0}` are the current learning 
rate of the current round and the initial learning rate respectively. :math:`F\left(\mathbf{x}_{t=l}\right)` and :math:`F\left(\mathbf{x}_{t=l}\right)` are
current training loss and initial loss in the first round respectively.


User-defined Step Size for LocalSGD
---------
There are three main parameters of the user-defined step size localsgd training mode, which are:

..  csv-table::
    :header: "Option", "Dtype", "Optional value", "Explain"
    :widths: 3, 3, 3, 5

    ":code:`use_local_sgd`", "bool", "False/True", "Whether to enable LocalSGD. Default: No"
    ":code:`local_sgd_is_warm_steps`", "int", "Greater than 0", "How many rounds of training before using local SGD"
    ":code:`local_sgd_steps`", "int", "Greater than 0", "Step size of LocalSGD"

Explain:

- LocalSGD's warmup step size :code:`local_sgd_is_warm_steps`  affects the generalization ability of the final model. Generally, you need to wait for the model parameters to stabilize before performing local SGD training. The experience value can use the epoch when the learning rate drops for the first time as the warmup step, and then start training for LocalSGD.
- LocalSGD's step size :code:`local_sgd_steps`. Generally, the larger the value, the less the communication times, and the faster the training speed, but the accuracy of the model decreases with it. The experience value is set to 2 or 4.

LocalSGD training can be achieved by setting the above three parameters, 
and only a few parts need to be added to the original distributed training code:

**Setting Up Distribution Strategy**

In the training strategy, select the :code:`use_local_sgd` switch.

.. code-block:: python

            dist_strategy = DistributedStrategy()
            # set LocalSGD mode
            dist_strategy.use_local_sgd = True


**Defining the Warmup Process**

A custom warmup strategy is required. 
In the previous :code:`local_sgd_warmup` process, 
normal SGD training is still used.

.. code-block:: python

            # defining the warmup process
            def get_local_sgd_steps(passid, local_sgd_steps, local_sgd_warmup):
                offset = passid - local_sgd_warmup
                if offset < 0:
                    return 1
                warm_up = [2 ** i for i in range(local_sgd_steps) if 2 ** i <=local_sgd_steps]
                if offset >= len(warm_up):
                    return local_sgd_steps
                else:
                    return warm_up[offset]

**Add Program Conversion Code**

There are two main programs here: one is :code:`fleet._origin_program` without the 
communication op, such as :code:`c_allreduce_sum`;
the other is :code:`fleet.main_program` that contains the communication op to perform distributed SGD training. 
This is mainly done in different :code:`local_sgd_steps` and switch 
between different programs to reduce communication times.

.. code-block:: python

            # get the local steps of the current round
            cur_local_sgd = get_local_sgd_steps(pass_id, local_sgd_steps, local_sgd_is_warm_steps)
            # switch different program according to step_cnt
            if step_cnt % cur_local_sgd == 0:
                current_prog = fleet.main_program
            else:
                current_prog = fleet._origin_program
            loss, acc1, acc5, lr = train_exe.run(current_prog, fetch_list=train_fetch_list, use_program_cache=True)


The complete training code of LocalSGD can refer to:
https://github.com/PaddlePaddle/Fleet/tree/develop/examples/local_sgd/resnet


Adaptive Step Size for LocalSGD
---------

The adaptive step size LocalSGD depends on the learning rate, 
so it is only applicable to optimization methods such as SGD 
that can obtain the global learning rate, 
but not to methods such as Adam. 
So compared to user-defined step size LocalSGD, 
the training method does not need to set the local step :code:`local_sgd_steps` and the warmup step 
:code:`local_sgd_is_warm_steps` .
Correspondingly, it is necessary to add code to obtain the current 
training loss and the current learning rate. 
The specific adding steps are as follows:

**Get the Current Training Loss**

Because the training loss of each card in distributed training is not consistent, 
it is necessary to synchronize the respective training losses at the end of each round.

.. code-block:: python

            # construct a net to synchronize train loss
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
            # initial the net
            all_train_prog = fluid.Program()
            all_startup_prog = fluid.Program()
            result = build_allreduce_program(all_train_prog, all_startup_prog)
            all_place = fluid.CUDAPlace(gpu_id)
            all_exe = fluid.Executor(all_place)
            all_exe.run(all_startup_prog)

**Get the Current Step Size Adaptively**

According to the current training loss, initial loss, 
current learning rate, and initial learning rate, the current training step is calculated.


.. code-block:: python

            # define adaptive acquisition training step size
            def adaptive_local_step(ini_loss, ini_lr, cur_loss, cur_lr, base_step, pre_step):
                # Reference: https://arxiv.org/pdf/1810.08313.pdf
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


**Add Program Conversion Code**

Similar to above.

.. code-block:: python

            # get training loss for current round
            all_loss = all_exe.run(all_train_prog,
                      feed={'tindata': train_loss},
                      fetch_list=[result.name])
            reduce_loss = float(all_loss[0]) / num_trainers
            # get current local step
            cur_local_sgd = adaptive_local_step(ini_loss, ini_lr, cur_loss, cur_lr, base_step, pre_step)
            # save the previous round of steps to prevent training fluctuations, 
            # resulting in drastic changes in local steps
            pre_step = cur_local_sgd
            # switch different program according to step_cnt
            if step_cnt % cur_local_sgd == 0:
                current_prog = fleet.main_program
            else:
                current_prog = fleet._origin_program
            loss, acc1, acc5, lr = train_exe.run(current_prog, fetch_list=train_fetch_list, use_program_cache=True)

The complete training code of LocalSGD can refer to:
https://github.com/PaddlePaddle/Fleet/tree/develop/examples/local_sgd/resnet


Reference
---------

[1] Lin T, Stich S U, Patel K K, et al. Don't Use Large Mini-Batches, Use Local SGD[J]. arXiv preprint arXiv:1808.07217, 2018.

[2] Wang J, Joshi G. Adaptive communication strategies to achieve the best error-runtime trade-off in local-update SGD[J]. arXiv preprint arXiv:1810.08313, 2018.


