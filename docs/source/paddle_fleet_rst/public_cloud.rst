公有云配置
---------------


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

