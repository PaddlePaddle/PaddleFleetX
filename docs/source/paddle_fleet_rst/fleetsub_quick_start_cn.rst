使用fleetsub提交集群任务
------------------------

``fleetsub``\ 是什么
~~~~~~~~~~~~~~~~~~~~

当您安装了FleeX后，便可以使用\ ``fleetsub``\ 在集群上提交分布式任务。长期的目标是成为集群任务提交的统一命令，只需要一行启动命令，就会将训练任务提交到离线训练集群中。目前该功能只支持百度公司内部云上的任务提交，使用fleetsub前需要先安装paddlecloud客户端，后续我们会支持更多的公有云任务提交。

使用要求
~~~~~~~~

使用\ ``fleetsub``\ 命令的要求：安装\ ``fleet-x``

-  【方法一】从pip源安装

.. code:: sh

       pip install fleet-x

-  【方法二】下载whl包并在本地安装

.. code:: sh

       # python2
       wget --no-check-certificate https://fleet.bj.bcebos.com/fleet_x-0.0.4-py2-none-any.whl
       pip install fleet_x-0.0.4-py2-none-any.whl
       # python3
       wget --no-check-certificate https://fleet.bj.bcebos.com/fleet_x-0.0.4-py3-none-any.whl
       pip3 install fleet_x-0.0.4-py3-none-any.whl

使用说明
~~~~~~~~

在提交任务前，用户需要在yaml文件中配置任务相关的信息，如：节点数、镜像地址、集群信息、启动训练所需要的命令等。

下面我们将为您详细讲解每个选项所代表的含义

任务提交
~~~~~~~~

定义完上述脚本后，用户即可使用\ ``fleetsub``\ 命令向PaddleCloud
提交任务了：

.. code:: sh

       fleetsub -f demo.yaml

使用样例
~~~~~~~~

具体的使用说明及样例代码请参考下面的\ `WIKI <http://wiki.baidu.com/pages/viewpage.action?pageId=1236728968>`__
