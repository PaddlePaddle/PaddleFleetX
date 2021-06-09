弹性训练
------------------

概述
^^^^^^^^^^^^^^^^^^^^^^

在分布式训练中，单节点故障导致这个任务失败的情况时有发生，尤其是节点较多的场景，不仅出错概率增加，重新运行任务的代价也相对更高。
针对这样的场景，弹性训练的需求应运而生。

paddle 目前已支持基于热重启的容错方案。

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

推荐通过 paddle-operator 使用该功能，详见 `kubernetes 部署 <https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/paddle_on_k8s.html>`_ .

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

3. 用户程序入口需要使用 fleetrun 启动，相关参数可通过环境变量或者启动参数提供

在环境中添加以下变量

.. code-block::

    PADDLE_ELASTIC_SERVER = 127.0.0.1:2379    # etcd 服务地址
    PADDLE_ELASTIC_FAULT_TOLERANC_LEVEL = 1   # 启用容错等级
    PADDLE_ELASTIC_JOB_ID = 'XXXXXX'          # 任务唯一id
    PADDLE_ELASTIC_NP = 2                     # 本次任务节点数
    POD_IP = 10.10.10.1                       # 本节点地址，一般由调度系统指定，或不配置由程序获取

或者使用以下命令运行 (注意需要在 np 个节点上都运行该命令)，

.. code-block::

    fleetrun --elastic_server=127.0.0.1:2379 --np=2 --job_id=XXXXXX --host=10.10.10.2 train_fleet_dygraph_ckpt.py


弹性
^^^^^^^^^^^^^^^^^^^^^^

弹性功能正在开发中。。。

