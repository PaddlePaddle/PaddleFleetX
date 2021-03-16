公有云配置
==============

公有云上有两类产品可以方便的运行 paddle，一是基于 kubernetes 的云原生容器引擎，例如百度云CCE产品、阿里云ACK产品、华为云CCE产品等；二是AI训练平台，例如百度云BML平台、华为云ModelArts平台、阿里云PAI平台。


1、在基于 kubernetes 的云原生容器引擎产品上使用 paddle
----

在公有云上运行 paddle 分布式可以通过选购容器引擎服务的方式，各大云厂商都推出了基于标准 kubernetes 的云产品，

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
* 部署 paddle-opeartor，详见下节；
* 提交 paddle 任务。

2、在AI训练平台产品上使用 paddle
----

2.1、百度云BML平台
^^^^^^^^


2.2、华为云ModelArts平台
^^^^^^^^


2.3、阿里云PAI平台
^^^^^^^^

由于阿里云PAI平台不支持自定义框架的方式来提交训练任务，目前 paddle 还无法在阿里云PAI平台上运行。