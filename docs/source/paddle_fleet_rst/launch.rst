启动分布式任务
------------------

飞桨通过 ``paddle.distributed.launch`` 组件启动分布式任务。该组件可用于启动单机多卡分布式任务，也可以用于启动多机多卡分布式任务。该组件为每张参与分布式任务的训练卡启动一个训练进程。默认情形下，该组件将在每个节点上启动 ``N`` 个进程，这里 ``N`` 等于训练节点的卡数，即使用所有的训训练卡。用户也可以通过 ``gpus`` 参数指定训练节点上使用的训练卡列表，该列表以逗号分隔。需要注意的是，所有节点需要使用相同数量的训练卡数。

为了启动多机分布式任务，需要通过 ``ips`` 参数指定所有节点的IP地址列表，该列表以逗号分隔。需要注意的是，该列表在所有节点上需要保持一致，即各节点IP地址出现的顺序需要保持一致。

例如，可以通过下面的命令启动单机多卡分布式任务：

.. code-block::

   python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 train_script

其中， ``train_script`` 为用户训练脚本。

从单机到多机分布式任务，只需额外指定 ``ips`` 参数即可。其内容为多机的IP列表，命令如下所示（假设两台机器的IP地址分别为192.168.0.1和192.168.0.2）：

.. code-block::
   
   python -m paddle.distributed.launch --ips="192.168.0.1,192.168.0.2" train_script

当省略 ``gpus`` 参数时，将根据训练节点上可用卡数启动相应数量的进程。用户也可使用 ``gpus`` 参数指定每个节点上使用的训练卡，命令如下所示：

.. code-block::
   
   python -m paddle.distributed.launch --gpus=0,1,2,3 --ips="192.168.0.1,192.168.0.2" train_script

下面将介绍 ``paddle.distributed.launch`` 组件在不同场景下的详细使用方法。

Collective分布式任务
~~~~~~~~~~~~~~~~~~~~~

Collective分布式任务场景下， ``paddle.distributed.launch`` 组件支持以下参数：

.. code-block::
   
   usage: launch.py [-h] [--log_dir LOG_DIR]
                    [--run_mode RUN_MODE] [--gpus GPUS] [--ips IPS]
                    training_script ...
   
   启动分布式任务 
   
   optional arguments:
     -h, --help            给出该帮助信息并退出
   
   Base Parameters:
     --log_dir LOG_DIR     训练日志的保存目录，默认：--log_dir=log/
     --run_mode RUN_MODE   任务运行模式, 可以为以下值: collective/ps/ps-heter；
                           当为collective模式时可省略。
     --gpus GPUS           训练使用的卡列表，以逗号分隔。例如: --gpus="4,5,6,7"
                           将使用节点上的4，5，6，7四张卡执行任务，并分别为每张卡
                           启动一个任务进程。
     training_script       用户的任务脚本，其后为该任务脚本的参数。
     training_script_args  用户任务脚本的参数
   
   Collective Parameters:
     --ips IPS             参与分布式任务的节点IP地址列表，以逗号分隔，例如：
                           192.168.0.16,192.168.0.17
   
各个参数的含义如下：

-  log_dir：训练日志储存目录。该目录下包含 ``endpoints.log`` 文件和各个卡的训练日志 ``workerlog.x`` （如workerlog.0，wokerlog.1等），其中 ``endpoints.log`` 文件记录各个训练进程的IP地址和端口号。
-  run_mode：运行模式，如collecitve，ps（parameter-server）或者ps-heter。
-  gpus：每个节点上使用的gpu卡的列表，以逗号间隔。例如 ``--gpus="0,1,2,3"``\ 。需要注意：这里的指定的卡号为物理卡号，而不是逻辑卡号。
-  training_script：训练脚本，如 ``train.py``\ 。
-  training_script_args：训练脚本的参数。
-  ips：所有训练节点的IP地址列表，以逗号间隔。例如， ``--ips="192.168.0.1,192.168.0.2``\ 。需要注意的是，该列表在所有节点上需要保持一致，即各节点IP地址出现的顺序在所有节点的任务脚本中需要保持一致。

PaddleCloud平台
===================

当在百度内部PaddleCloud平台使用飞桨分布式时，可以省略 ``ips`` 参数。更多关于如何通过
在PaddleCloud上启动分布式任务，请参考PaddleCloud官方文档。