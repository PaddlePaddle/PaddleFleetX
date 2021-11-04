
Docker 镜像
---------------

概述
^^^^^^^^^^^^^^^^^^^^^^
本节详细描述 ParameterServer(PS) 训练示例 wide_and_deep 和 Collective 训练示例 resnet 的任务镜像构建过程， 
用户可以本节提供的基础镜像或方法

* 构建自己的镜像用于开发环境进行代码测试；
* 封装构建任务镜像用于在 kubernetes 等环境中执行训练任务。

环境需求：

* docker，建议 version 19+ ；
* docker 仓库，自建、云仓库（如百度云 `ccr <https://cloud.baidu.com/doc/CCR/s/qk8gwqs4a>`_ ）、 `dockerhub <https://hub.docker.com/>`_ 。

注意：本地开发不需要 docker 仓库，在 kubernetes 中使用镜像或镜像共享需要有可访问的镜像仓库，
本文使用百度云镜像仓库 ccr registry.baidubce.com/paddle-operator 作为示例，需要进行 `登录 <https://docs.docker.com/engine/reference/commandline/login/>`_ 。


示例 wide and deep
^^^^^^^^^^^^^^^^^^^^^^

本示例采用 PS 模式，使用 cpu 进行训练。

代码准备
============

示例源码可在此获得: `wide_and_deep <https://github.com/PaddlePaddle/FleetX/tree/develop/examples/wide_and_deep>`_ ，其中 train.py 为程序的入口点。

本示例会在任务镜像中包含训练数据，实际应用过程中一般不会也不建议这样使用，常见用法分为以下两种：

* 任务运行时，程序通过网络拉取数据到本地进行训练。该情形数据由程序维护，不需要额外配置；
* 任务运行时，程序读取本地目录进行训练，该情形需要使用用户配置 kubernetes 支持的挂载存储，一般建议使用 pvc 抽象，详细示例见 kubernetes 部署章节。 

制作任务镜像
============

用于生成镜像的 Dockerfile 和代码目录，

.. code-block::

    $ ls
    Dockerfile   wide_and_deep

Dockerfile 内容，

.. code-block::

    $ cat Dockerfile
    FROM ubuntu:18.04

    RUN apt update && \
        apt install -y python3 python3-dev python3-pip
    
    RUN python3 -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple
    
    ## 以下根据用户内容修改

    ADD wide_and_deep /wide_and_deep
    
    WORKDIR /wide_and_deep
    
    ENTRYPOINT ["python3", "train.py"]

用户可根据实际情况更改安装内容（如 paddlepaddle 版本）和安装额外依赖。

制作镜像

.. code-block::

    docker build -t registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1 .

提交镜像 (需要具有对应权限)

.. code-block::

    docker push registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1



示例 resnet
^^^^^^^^^^^^^^^^^^^^^^

本示例采用 Collective 模式，使用 gpu 进行训练。

注意：

* 使用 gpu 训练时需要在集群中安装好对应 `驱动 <https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver>`_ 和  `工具包 <https://github.com/NVIDIA/nvidia-docker/blob/master/README.md#quickstart>`_ 支持;
* 封装镜像所选用的 cuda 版本需要和运行环境所安装的 cuda 版本对应。可通过 *nvidia-smi* 命令查看驱动和 cuda 版本。

代码准备
============

示例源码可在此获得: `resnet <https://github.com/PaddlePaddle/FleetX/tree/develop/examples/resnet>`_  ，其中 train_fleet.py 为程序的入口点。

制作任务镜像
============

用于生成镜像的 Dockerfile 和代码目录，

.. code-block::

    $ ls
    Dockerfile   resnet

Dockerfile 内容，

.. code-block::

    $ cat Dockerfile

    FROM registry.baidubce.com/paddle-operator/paddle-base-gpu:cuda10.2-cudnn7-devel-ubuntu18.04
    
    ADD resnet /resnet
    
    WORKDIR /resnet
    
    RUN pip install scipy opencv-python==4.2.0.32 -i https://mirror.baidu.com/pypi/simple
    
    CMD ["python","-m","paddle.distributed.launch","train_fleet.py"]

注意：

* 这里选用的 base 镜像为预装 paddle 的 gpu 镜像，制作过程见下一节，用户可根据 cuda 和所需版本选用；
* 用户可根据实际情况更改内容和安装额外依赖；
* 启动命令需要调用 paddle.distributed.launch 模块，具体信息参考对应章节。

制作镜像

.. code-block::

    docker build -t registry.baidubce.com/paddle-operator/demo-resnet:v1 .

提交镜像 (需要具有对应权限)

.. code-block::

    docker push registry.baidubce.com/paddle-operator/demo-resnet:v1


开发镜像
^^^^^^^^^^^^^^^^^^^^^^

本小节介绍使用 docker 环境镜像代码开发和调试环境的镜像构建，
以及上述例子中使用的发布环境的镜像构建。

使用 docker 环境作为开发环境的好处：

* 对环境进行封装，在不同机器上开发时保持环境一致，同时方便合作共享；
* 降低从开发到发布的 gap，降低发布成本。

本节涉及的 dockerfile 可以在 `这里 <https://github.com/PaddlePaddle/FleetX/tree/develop/dockerfiles>`_ 找到，

.. list-table::

  * - 镜像
    - 描述
    - 使用
    - Dockerfile
  * - registry.baidubce.com/paddle-operator/paddle-dev-env:2.0.0
    - cpu 镜像、paddle2.0.0、git/vim/curl
    - 推荐用于开发调试
    - cpu.2.0.0.Dockerfile
  * - registry.baidubce.com/paddle-operator/paddle-dev-env:1.8.5
    - cpu 镜像，paddle1.8.5、paddlerec/paddle-serving、git/vim/curl
    - 推荐用于开发调试
    - cpu.1.8.5.Dockerfile
  * - registry.baidubce.com/paddle-operator/paddle-base-gpu:11.2.1-cudnn8-devel-ubuntu18.04
    - gpu镜像、paddle2.0.0
    - 用于发布训练任务的基础环境
    - gpu.Dockerfile
  * - registry.baidubce.com/paddle-operator/paddle-base-gpu:10.2-cudnn7-devel-ubuntu18.04 
    - gpu镜像、paddle2.0.0
    - 用于发布训练任务的基础环境
    - gpu.Dockerfile
  * - registry.baidubce.com/paddle-operator/paddle-dev-env-gpu:2.0.0-gpu-cuda10.2-cudnn7
    - gpu镜像、paddle2.0.0、编译环境
    - 推荐用于开发调试
    - gpu.2.0.0.Dockerfile


通用开发环境推荐以 `官方镜像 <https://www.paddlepaddle.org.cn/>`_ 为 base 添加常用工具镜像构建，
该镜像提供运行和编译 paddle 依赖和多版本的 python 环境。

用户也可根据表中镜像选择适合或类似的镜像直接拉取使用或者根据 dokcerfile 进行定制，注意：

首先确定 cpu 或者 gpu 版本，如果是 gpu 版本需要确定 cuda 版本，详见 `nvidia/cuda  <https://hub.docker.com/r/nvidia/cuda>`_ 。

定制镜像命令参考

.. code-block::

  docker build --build-arg CUDA=11.2.1-cudnn8-devel-ubuntu18.04 -t registry.baidubce.com/paddle-operator/paddle-base-gpu:11.2.1-cudnn8-devel-ubuntu18.04 -f gpu.Dockerfile .

.. code-block::

  docker build --build-arg PADDLE_IMG=registry.baidubce.com/paddlepaddle/paddle:2.0.0-gpu-cuda11.0-cudnn8  -t registry.baidubce.com/paddle-operator/paddle-dev-env-gpu:2.0.0-gpu-cuda11.0-cudnn8 -f gpu.2.0.0.Dockerfile .


简单的 CPU 测试运行环境可以直接依赖 ubuntu 18.04 为基础镜像进行构建，特殊需求请根据实际更改。

发布镜像准备完毕后，可在发布前进行本地 docker 环境中进行调试，例如

.. code-block::

    docker run -it --entrypoint bash registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1
    python -m paddle.distributed.launch --server_num=1 --worker_num=2 train.py

飞桨官方镜像
^^^^^^^^^^^^^^^^^^^^^^

除了依据上述方法制作自定义开发镜像，用户也可以在 `DockerHub <https://hub.docker.com/r/paddlepaddle/paddle/tags/>`_ 中找到飞桨各个发行版本的官方docker镜像。