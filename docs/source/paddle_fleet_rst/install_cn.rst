安装PaddlePaddle
------------------

docker镜像安装
~~~~~~~~~~~~~

使用飞桨进行分布式训练的最小安装集合就是安装PaddlePaddle。我们强烈建议您通过飞桨官方docker镜像使用PaddlePaddle。飞桨官方镜像中包含已经安装最新发行版PaddlePaddle和相关的环境，如CUDA，CUDNN和NCCL等。关于如何获取官方docker镜像，请参考\ `安装信息 <https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html>`__\ 。更多关于docker的使用信息，请参考\ `docker文档 <https://docs.docker.com/>`__\ 和\ `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__\ 。在使用飞桨分布式时，请确保您在所有机器上部署了docker镜像。

物理机安装
~~~~~~~~~

当您选择在物理机安装PaddlePaddle，请确保您的物理机上安装了符合\ `安装指南 <https://www.paddlepaddle.org.cn/install/quick>`__\ 中\ **环境准备**\ 和\ **开始安装**\ 部分要求的操作系统、Python版本和CUDA工具包等，并在安装完后验证安装。需要注意的是，当您选择使用GPU进行分布式训练时，您还需要额外安装NVIDIA NCCL通信库。关于如何安装NVIDIA NCCL通信库，请参考\ `NCCL安装指南 <https://github.com/NVIDIA/nccl>`__\ 。在使用飞桨分布式时，请确保您在所有机器上安装了PaddlePaddle和上述软件环境。

K8S平台安装
~~~~~~~~~~

当您选择在K8S平台安装PaddlePaddle，请参考\ `Kubernetes 部署 <./paddle_on_k8s.html>`__\ 。

获取更多安装信息，请参考\ `安装指南 <https://www.paddlepaddle.org.cn/install/quick>`__\ 。

备注：**目前飞桨分布式仅支持Linux系统(CentOS/Ubuntu等)， 暂不支持Windows和Mac系统**


更多whl包下载
~~~~~~~~~~~~~~~~~~

您也可以自行选择下载PaddlePaddle whl安装版安装需要的PaddlePaddle版本。在安装前，请参考上面的章节安装docker镜像或者是需要的软件环境。更多whl包下载地址如下：

-  官方正式版， 可以从\ `多版本whl包列表-Release <https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release>`__\ 下载。
-  官方开发版， 可以从\ `多版本whl包列表-develop <https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-dev>`__\ 下载。

