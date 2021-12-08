弹性训练
------------------

概述
^^^^^^^^^^^^^^^^^^^^^^

在分布式训练中，单节点故障导致这个任务失败的情况时有发生，尤其是节点较多的场景，不仅出错概率增加，重新运行任务的代价也相对更高。
针对这样的场景，弹性训练的需求应运而生。

paddle 目前已支持 Collective 训练模式基于热重启的容错方案。


*热重启即用户的任务进程会被重启，所以需要用户代码中做好 checkpoint 逻辑。*

容错
^^^^^^^^^^^^^^^^^^^^^^

方案概述
~~~~~~~~~~~~~~~~~~~~~~~~

当前方案的容错是 pod 级别，即组成分布式网络节点为单位，该节点可能管理多卡，当该节点中出现故障时，该节点所有资源释放，需要重启。

基本设计思路包括以下几点：

* 使用中心化的外部服务 etcd 进行节点数据同步和异常节点的感知，同时无需调度平台收集所有节点信息进行配置；
* 当故障节点故障退出后，需要调度平台将新节点以同样配置（可能异地）重启后进入恢复训练阶段；
* 故障节点退出后，非故障节点感知到即进入无限等待模式，退出和超时由外部调度系统负责；
* 训练的恢复依赖在用户代码中设置的 checkpoint 完成;

使用方法
~~~~~~~~~~~~~~~~~~~~~~~~

推荐通过 paddle-operator 使用该功能，
在提交任务时指定需要开启弹性功能，

.. code-block::

    apiVersion: batch.paddlepaddle.org/v1
    kind: PaddleJob
    metadata:
      name: job-elastic
    spec:
      elastic: 1
    ...

详见 `kubernetes 部署 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/paddle_on_k8s.html>`_ .

以下通过 resnet 示例介绍使用方法：

0. 需要先安装 etcd server，以获得服务地址和端口如 127.0.0.1:2379

1. 用户程序运行环境中需要安装 etcd 客户端 etcd3 

.. code-block::

    pip install --no-cache-dir -U etcd3 -i https://mirror.baidu.com/pypi/simple

2. 在用户程序中添加 checkpoint 相关逻辑，主要部分如下

load 环节

.. code-block::

    resnet = ResNet(class_dim=class_dim, layers=50)
    if int(fleet.local_rank()) == 0 and checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            chkpt = paddle.load(checkpoint_path)
            resnet.set_state_dict(chkpt)
            start_epoch = chkpt.get('epoch',0)
            print("load checkpoint succuss")
        except Exception as e:
            print("load checkpoint failed", e)

save 环节

.. code-block::

    if int(fleet.local_rank()) == 0:
        state_dict = resnet.state_dict()
        state_dict['epoch'] = eop
        paddle.save(state_dict, checkpoint_path)

完整示例见 `resnet <https://github.com/PaddlePaddle/FleetX/tree/develop/examples/resnet/train_fleet_dygraph_ckpt.py>`_ .

注意：如示例中所示，save 和 load 均在 rank 为 0 的节点上进行，checkpoint 所在目录要确保能够被访问。

3. 用户程序入口需要使用 python -m paddle.distributed.launch 启动，相关参数可通过环境变量或者启动参数提供

在环境中添加以下变量

.. code-block::

    PADDLE_ELASTIC_SERVER = 127.0.0.1:2379    # etcd 服务地址
    PADDLE_ELASTIC_FAULT_TOLERANC_LEVEL = 1   # 启用容错等级
    PADDLE_ELASTIC_JOB_ID = 'XXXXXX'          # 任务唯一id
    PADDLE_ELASTIC_NP = 2                     # 本次任务节点数
    POD_IP = 10.10.10.1                       # 本节点地址，一般由调度系统指定，或不配置由程序获取

或者使用以下命令运行 (注意需要在 np 个节点上都运行该命令)，

.. code-block::

    python -m paddle.distributed.launch --elastic_server=127.0.0.1:2379 --np=2 --job_id=XXXXXX train_fleet_dygraph_ckpt.py


弹性
^^^^^^^^^^^^^^^^^^^^^^

概述
^^^^^^^^^^^^^^^^^^^^^^

在分布式训练中，除了容错外，集群的资源剩余情况可能随时间而不同、任务的优先级也可能有不同，
基于这样的场景，实现弹性训练即任务可以在运行时动态调整训练资源而不影响或尽可能小地影响训练进程，能够最大限度地实现资源利用率提升同时提升训练任务质量。

paddle 目前已支持 Collective 训练模式基于热重启的弹性训练方案。

*热重启即用户的任务进程会被重启，所以需要用户代码中做好 checkpoint 逻辑，同时如 batchsize 和 learning rate 这样需要随节点数变化的参数也需要用户进程自动调整。*

方案概述
~~~~~~~~~~~~~~~~~~~~~~~~

本方案以上述容错为基础，分为扩容和缩容两种情况，流程如下：

1. 正常启动训练；
2. 为训练任务配置可变节点数，例如np=2:4，表示任务最小需要2个节点，最大需要4个节点，平台调度系统可以在这个区间内为任务分配计算节点；
3. 训练过程中，若为任务扩容，当节点加入满足所需训练节点要求时，各节点感知判断满足训练条件，继续训练； 若为任务缩容，当节点剩余节点满足所需训练节点要求时，各节点感知判断满足训练条件，继续训练；

> 这里的节点加入和退出需要外部调度系统负责实现。

使用方法
~~~~~~~~~~~~~~~~~~~~~~~~

推荐通过 paddle-operator 使用该功能，首先在提交任务中开启弹性功能，然后任务正常运行中通过 kubectl 或 api 的其他方式修改 paddlejob 中的 replicas 字段即可实现改功能。
详见 `kubernetes 部署 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/paddle_on_k8s.html>`_ .

以下通过 resnet 示例介绍使用弹性的方法：

1. 运行任务 (注意需要在 np 个节点上都运行该命令)，

.. code-block::

    python -m paddle.distributed.launch --elastic_server=127.0.0.1:2379 --np=2:4 --job_id=XXXXXX train_fleet_dygraph_ckpt.py

2. 执行扩容或缩容

通过k8s进行扩缩容操作，下面的命令是执行扩容操作，由2个节点扩容到3个节点（缩容也类似），等待超时时间可由PADDLE_ELASTIC_TIMEOUT（默认值是120秒）环境变量控制

.. code-block::
    
    kubectl scale --current-replicas=2 --replicas=3 paddlejob/paddle-controller-manager-698dd7b855-n65jr
    

