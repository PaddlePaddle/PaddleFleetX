
Kubernetes 部署
---------------

概述
^^^^^^^^^^^^^^^^^^^^^^

在 kubernetes 上部署分布式任务需要安装 `paddle-operator <https://github.com/PaddleFlow/paddle-operator>`_ 。
paddle-operator 通过添加自定义资源类型 (paddlejob) 以及部署 controller 和一系列 kubernetes 原生组件的方式实现简单定义即可运行 paddle 任务的需求。

目前支持运行 ParameterServer (PS) 和 Collective 两种分布式任务，当然也支持运行单节点任务。

paddle-operator 安装
^^^^^^^^

准备
~~~~~~

安装 paddle-operator 需要有已经安装的 kubernetes (v1.8+) 集群和 `kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl/>`_  (v1.8+) 工具。

本节所需配置文件和示例可以在 `这里 <https://github.com/PaddleFlow/paddle-operator/tree/main/deploy>`_ 找到，
可以通过 *git clone* 或者复制文件内容保存。

.. code-block::

    deploy
    |-- examples
    |   |-- resnet.yaml
    |   |-- wide_and_deep.yaml
    |   |-- wide_and_deep_podip.yaml
    |   |-- wide_and_deep_service.yaml
    |   `-- wide_and_deep_volcano.yaml
    |-- v1
    |   |-- crd.yaml
    |   `-- operator.yaml
    `-- v1beta1
        |-- crd.yaml
        `-- operator.yaml


部署 CRD
~~~~~~~~~~~~~~~~~~~~~~~~

*注意：kubernetes 1.15 及以下使用 v1beta1 目录，1.16 及以上使用目录 v1.*

执行以下命令，

.. code-block::

   $ kubectl create -f https://raw.githubusercontent.com/PaddleFlow/paddle-operator/dev/deploy/v1/crd.yaml

或者

.. code-block::

   $ kubectl create -f deploy/v1/crd.yaml

*注意：v1beta1 请根据报错信息添加 --validate=false 选项*

通过以下命令查看是否成功，

.. code-block::

    $ kubectl get crd
    NAME                                    CREATED AT
    paddlejobs.batch.paddlepaddle.org       2021-02-08T07:43:24Z
 
部署 controller 及相关组件
~~~~~~~~~~~~~~~~~~~~~~~~

*注意：默认部署的 namespace 为 paddle-system，如果希望在自定义的 namespace 中运行或者提交任务，
需要先在 operator.yaml 文件中对应更改 namespace 配置，其中*

* *namespace: paddle-system* 表示该资源部署的 namespace，可理解为系统 controller namespace；
* Deployment 资源中 containers.args 中 *--namespace=paddle-system* 表示 controller 监控资源所在 namespace，即任务提交 namespace。


执行以下部署命令，

.. code-block::

   $ kubectl create -f https://raw.githubusercontent.com/PaddleFlow/paddle-operator/dev/deploy/v1/operator.yaml

或者

.. code-block::

   $ kubectl create -f deploy/v1/operator.yaml

通过以下命令查看部署结果和运行状态，

.. code-block::

    $ kubectl -n paddle-system get pods
    NAME                                         READY   STATUS    RESTARTS   AGE
    paddle-controller-manager-698dd7b855-n65jr   1/1     Running   0          1m

通过查看 controller 日志以确保运行正常，

.. code-block::

    $ kubectl -n paddle-system logs paddle-controller-manager-698dd7b855-n65jr

提交 demo 任务查看效果，

.. code-block::

   $ kubectl -n paddle-system create -f deploy/examples/wide_and_deep.yaml

查看 paddlejob 任务状态, pdj 为 paddlejob 的缩写，

.. code-block::

    $ kubectl -n paddle-system get pdj
    NAME                     STATUS      MODE   PS    WORKER   AGE
    wide-ande-deep-service   Completed   PS     2/2   0/2      4m4s

以上信息可以看出：训练任务已经正确完成，该任务为 ps 模式，配置需求 2 个 pserver, 2 个在运行，需求 2 个 woker，0 个在运行（已完成退出）。
可通过 cleanPodPolicy 配置任务完成/失败后的 pod 删除策略，详见任务配置。

查看 pod 状态，

.. code-block::

   $ kubectl -n paddle-system get pods

卸载
~~~~~~

通过以下命令卸载部署的组件，

.. code-block::

   $ kubectl delete -f deploy/v1/crd.yaml -f deploy/v1/operator.yaml

*注意：重新安装时，建议先卸载再安装*


paddlejob 任务提交
^^^^^^^^

在上述安装过程中，我们使用了 wide-and-deep 的例子作为提交任务演示，本节详细描述任务配置和提交流程供用户参考提交自己的任务，
镜像的制作过程可在 *docker 镜像* 章节找到。

示例 wide and deep
~~~~~~~~~~~~~~~~~~~~~~~~

本示例采用 PS 模式，使用 cpu 进行训练，所以需要配置 ps 和 worker。

准备配置文件，

.. code-block::
    
    $ cat demo-wide-and-deep.yaml
    apiVersion: batch.paddlepaddle.org/v1
    kind: PaddleJob
    metadata:
      name: wide-ande-deep
    spec:
      withGloo: 1
      intranet: PodIP
      cleanPodPolicy: OnCompletion
      worker:
        replicas: 2
        template:
          spec:
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1
      ps:
        replicas: 2
        template:
          spec:
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1

说明：

* 提交命名需要唯一，如果存在冲突请先删除原 paddlejob 确保已经删除再提交;
* ps 模式时需要同时配置 ps 和 worker，collective 模式时只需要配置 worker 即可；
* withGloo 可选配置为 0 不启用， 1 只启动 worker 端， 2 启动全部(worker端和Server端)， 建议设置 1；
* cleanPodPolicy 可选配置为 Always/Never/OnFailure/OnCompletion，表示任务终止（失败或成功）时，是否删除 pod，调试时建议 Never，生产时建议 OnCompletion；
* intranet 可选配置为 Service/PodIP，表示 pod 间的通信方式，用户可以不配置, 默认使用 PodIP；
* ps 和 worker 的内容为 podTemplateSpec，用户可根据需要遵从 kubernetes 规范添加更多内容, 如 GPU 的配置.


提交任务: 使用 kubectl 提交 yaml 配置文件以创建任务，

.. code-block::
    
    $ kubectl -n paddle-system create -f demo-wide-and-deep.yaml

示例 resnet
~~~~~~~~~~~~~~~~~~~~~~~~

本示例采用 Collective 模式，使用 gpu 进行训练，所以只需要配置 worker，且需要配置 gpu。

准备配置文件，

.. code-block::

    $ cat resnet.yaml
    apiVersion: batch.paddlepaddle.org/v1
    kind: PaddleJob
    metadata:
      name: resnet
    spec:
      cleanPodPolicy: Never
      worker:
        replicas: 2
        template:
          spec:
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/demo-resnet:v1
                command:
                - python
                args:
                - "-m"
                - "paddle.distributed.launch"
                - "train_fleet.py"
                volumeMounts:
                - mountPath: /dev/shm
                  name: dshm
                resources:
                  limits:
                    nvidia.com/gpu: 1
            volumes:
            - name: dshm
              emptyDir:
                medium: Memory
        

注意：

* 这里需要添加 shared memory 挂载以防止缓存出错；
* 本示例采用内置 flower 数据集，程序启动后会进行下载，根据网络环境可能等待较长时间。

提交任务: 使用 kubectl 提交 yaml 配置文件以创建任务，

.. code-block::
    
    $ kubectl -n paddle-system create -f resnet.yaml

更多配置
^^^^^^^^

Volcano 支持
~~~~~~~~~~~~~~~~~~~~~~~~

paddle-operator 支持使用 volcano 进行复杂任务调度，使用前请先 `安装 <https://github.com/volcano-sh/volcano>`_ 。

本节使用 volcano 实现 paddlejob 运行的 gan-scheduling。

使用此功能需要进行如下配置：

* 创建 paddlejob 同名 podgroup，具体配置信息参考 volcano 规范；
* 在 paddlejob 任务配置中添加声明：schedulerName: volcano , 注意：需要且只需要在 worker 中配置。

配置示例，

.. code-block::

    ---
    apiVersion: batch.paddlepaddle.org/v1
    kind: PaddleJob
    metadata:
      name: wide-ande-deep
    spec:
      cleanPodPolicy: Never
      withGloo: 1
      worker:
        replicas: 2
        template:
          spec:
            restartPolicy: "Never"
            schedulerName: volcano
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1
      ps:
        replicas: 2
        template:
          spec:
            restartPolicy: "Never"
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1
    
    ---
    apiVersion: scheduling.volcano.sh/v1beta1
    kind: PodGroup
    metadata:
      name: wide-ande-deep
    spec:
      minMember: 4

在以上配置中，我们通过创建最小调度单元为 4 的 podgroup，并将 paddlejob 任务标记使用 volcano 调度，实现了任务的 gan-scheduling。

可以通过以下命运提交上述任务查看结果，

.. code-block::

   $ kubectl -n paddle-system create -f deploy/examples/wide_and_deep.yaml


GPU 和节点选择
~~~~~~~~~~~~~~~~~~~~~~~~

更多配置示例，

.. code-block::

    apiVersion: batch.paddlepaddle.org/v1
    kind: PaddleJob
    metadata:
      name: wide-ande-deep
    spec:
      intranet: Service
      cleanPodPolicy: OnCompletion
      worker:
        replicas: 2
        template:
          spec:
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1
                resources:
                  limits:
                    nvidia.com/gpu: 1
            nodeSelector:
              accelerator: nvidia-tesla-p100
      ps:
        replicas: 2
        template:
          spec:
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1
                resources:
                  limits:
                    nvidia.com/gpu: 1
            nodeSelector:
              accelerator: nvidia-tesla-p100

数据存储
~~~~~~~~~~~~~~~~~~~~~~~~

在 kubernentes 中使用挂载存储建议使用 pv/pvc 配置，详见 `persistent-volumes <https://kubernetes.io/docs/concepts/storage/persistent-volumes/>`_ 。

这里使用 nfs 云盘作为存储作为示例，配置文件如下，

.. code-block::

    $ cat pv-pvc.yaml
    ---
    apiVersion: v1
    kind: PersistentVolume
    metadata:
      name: nfs-pv
    spec:
      capacity:
        storage: 10Gi
      volumeMode: Filesystem
      accessModes:
        - ReadWriteOnce
      persistentVolumeReclaimPolicy: Recycle
      storageClassName: slow
      mountOptions:
        - hard
        - nfsvers=4.1
      nfs:
        path: /nas
        server: 10.12.201.xx
    
    ---
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: nfs-pvc
    spec:
      accessModes:
        - ReadWriteOnce
      volumeMode: Filesystem
      resources:
        requests:
          storage: 10Gi
      storageClassName: slow
      volumeName: nfs-pv
    

使用以下命令在 namespace paddle-system 中  创建 pvc 名为 nfs-pvc 的存储声明，实际引用为 10.12.201.xx 上的 nfs 存储。

.. code-block::

   $ kubectl -n paddle-system apply -f pv-pvc.yaml
    
注意 pvc 需要绑定 namespace 且只能在该 namespace 下使用。
    
提交 paddlejob 任务时，配置 volumes 引用以使用对应存储，

.. code-block::

    apiVersion: batch.paddlepaddle.org/v1
    kind: PaddleJob
    metadata:
      name: paddlejob-demo-1
    spec:
      cleanPolicy: OnCompletion
      worker:
        replicas: 2
        template:
          spec:
            restartPolicy: "Never"
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/paddle-ubuntu:2.0.0-18.04
                command: ["bash","-c"]
                args: ["cd /nas/wide_and_deep; python3 train.py"]
                volumeMounts:
                - mountPath: /nas
                  name: data
            volumes:
              - name: data
                persistentVolumeClaim:
                  claimName: nfs-pvc
      ps:
        replicas: 2
        template:
          spec:
            restartPolicy: "Never"
            containers:
              - name: paddle
                image: registry.baidubce.com/paddle-operator/paddle-ubuntu:2.0.0-18.04
                command: ["bash","-c"]
                args: ["cd /nas/wide_and_deep; python3 train.py"]
                volumeMounts:
                - mountPath: /nas
                  name: data
            volumes:
              - name: data
                persistentVolumeClaim:
                  claimName: nfs-pvc

该示例中，镜像仅提供运行环境，训练代码和数据均通过存储挂载的方式添加。

