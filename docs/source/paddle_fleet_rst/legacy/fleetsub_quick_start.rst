使用fleetsub提交集群任务
------------------------

Fleetsub是什么
~~~~~~~~~~~~~~

当您安装了FleetX后，便可以使用\ ``fleetsub``\在云上提交分布式任务。只需要一行启动命令，就会将训练任务提交到PaddleCloud进行多机多卡分布式训练。

目前该功能只支持百度公司内部云上的任务提交，使用fleetsub前需要先安装paddlecloud客户端，后续我们会支持更多的公有云任务提交。

使用要求
~~~~~~~~

使用\ ``fleetsub``\ 命令的要求：安装\ ``fleetx``\

【方法一】从pip源安装

.. code:: sh 

   pip install fleet-x

【方法二】下载whl包并在本地安装

.. code:: sh

   # python2
   wget --no-check-certificate https://fleet.bj.bcebos.com/fleet_x-0.0.4-py2-none-any.whl
   pip install fleet_x-0.0.4-py2-none-any.whl
   # python3
   wget --no-check-certificate https://fleet.bj.bcebos.com/fleet_x-0.0.4-py3-none-any.whl
   pip3 install fleet_x-0.0.4-py3-none-any.whl

使用说明
~~~~~~~


在提交任务前，用户需要在yaml文件中配置任务相关的信息，如：机器数、镜像地址、集群信息、启动训练所需要的命令等。

下面我们将为您详细讲解每个选项所代表的含义（文件名称demo.yaml）：

任务配置相关
^^^^^^^^^^^^

- **num_trainers:** 训练中所用到的机器数。一般为2的指数，如：1、2、4、8等

- **num_cards:** 每台机器所用到的GPU卡数。一般为2的指数，但最大为8（1、2、4、8）。该参数与\ ``num_trainers``\ 相乘即为训练中所用到的所有GPU卡。

- **job_prefix:** 任务的前缀。配合机器数及每台机器的卡数，会为用户生成任务的名字。如：任务的前缀为"test"，训练中使用单机八卡，则提交的任务名称为: "test_N1C8"。

- **image_addr:** 训练中使用的镜像地址。我们为用户提供了不同python版本的\ `镜像 <镜像链接>`_

- **group_name:** 机器群组的名称。

- **cluster_name:** 集群类型。一般指训练中所用到的GPU类型（V100，P40等）。

- **server:** PaddleCloud服务地址，通常不需要用户配置。

- **over_sell:** 是否是用超发。若开启超发，当用户指定的集群资源已经被占满时，会借用其他集群空闲的集群运行任务。但开启超发时，任务会因为GPU利用率不高等原因被终止。所以用户需要根据自己的需求决定是否使用超发。

- **log_fs_name:** 任务中的日志最终会被传送到afs集群，以便用户以后查找，该选项需要用户指定afs集群的地址。

- **log_fs_ugi:** 为了能成功访问上面定义的afs地址，用户还需要提供afs上的用户信息。

- **log_output_path:** 最后用户需要定义日志被传送的路径。

任务执行相关
^^^^^^^^^^^^

- **upload_files:** 需要上传的文件。如：训练脚本、数据下载相关配置文件等。

- **whl_install_commands:** 在镜像中安装额外的whl包所需要的指令。在任务执行中，用户可能会需要安装额外的whl包，或更新镜像中已有功能的版本。在这个选项中，用户可以定义安装/更新功能版本的全部命令。

- **commands：** 训练启动命令。在准备完机器及环境所有的配置后，用户就可以定义启动分布式训练任务所需要的命令了，如：\ ``fleetrun xxx.py``\。


任务提交
^^^^^^^^

定义完上述脚本后，用户即可使用\ ``fleetsub`` \命令向PaddleCloud 提交任务了：

.. code:: sh

   fleetsub -f demo.yaml

使用样例
~~~~~~~

具体的使用说明及样例代码请参考下面的\ `链接 <wiki>`_
