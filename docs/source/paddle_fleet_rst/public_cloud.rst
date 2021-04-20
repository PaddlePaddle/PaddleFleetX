公有云配置
==============

公有云上有两类产品可以方便的运行 PaddlePaddle，一是基于 kubernetes 的云原生容器引擎，例如百度云CCE产品、阿里云ACK产品、华为云CCE产品等；二是各个云厂商的AI开发平台，例如百度云BML平台、华为云ModelArts平台、阿里云PAI平台。


1、在基于 kubernetes 的云原生容器引擎产品上使用 PaddlePaddle
----

在公有云上运行 PaddlePaddle 分布式可以通过选购容器引擎服务的方式，各大云厂商都推出了基于标准 kubernetes 的云产品，

.. list-table::
  
  * - 云厂商
    - 容器引擎
    - 链接
  * - 百度云
    - CCE
    - https://cloud.baidu.com/product/cce.html
  * - 阿里云
    - ACK
    - https://help.aliyun.com/product/85222.html
  * - 华为云
    - CCE
    - https://www.huaweicloud.com/product/cce.html

使用流程：

* 购买服务，包括节点及 cpu 或 gpu 计算资源；

* \ `构建Docker镜像 <./docker_image.html>`__\、\ `kubernetes部署paddle-opeartor <./paddle_on_k8s.html>`__\；
* \ `提交PaddlePaddle任务 <./paddle_on_k8s.html#paddlejob>`__\。 

2、在云厂商AI开发平台产品上使用 PaddlePaddle
----

.. toctree::
   :maxdepth: 2 
   :name: cloud

   cloud/bml_guide.rst 
   cloud/modelarts_guide.rst
   cloud/aliyun_pai_guide.rst

