启动分布式任务
------------------

飞桨通过 ``paddle.distributed.launch`` 组件启动分布式任务。该组件可用于启动
单机单卡任务，也可以用于启动单机多卡和多机多卡分布式任务。该组件为每张训练卡
启动一个训练进程。默认情形下，该组件将在每个节点上启动 ``N`` 个进程，
这里 ``N`` 等于训练节点的卡数。用户也可以通过 ``gpus`` 参数指定训练节点上使用
的训练卡列表，该列表以逗号分隔。需要注意的是，所有节点需要使用相同数量的训练
卡数。

为了启动多机分布式任务，需要通过 ``ips`` 参数指定所有节点的ip地址列表，该列表
以逗号分隔。需要注意的是，所有该列表在所有节点上需要保持一致。

例如，可以通过下面的命令启动单机任务：

.. code-block::
   python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 train_script

从单机到多机分布式任务，只需再额外指定 ``ips`` 参数即可。其内容为多机的ip列表，命令如下所示（假设两台机器的ip地址分别为192.168.0.1和192.168.0.2）：

.. code-block::
   python -m paddle.distributed.launch --ips="192.168.0.1,192.168.0.2" train_script

下面将介绍 ``paddle.distributed.launch`` 组件在不同场景下的详细使用方法。

Collective分布式任务
~~~~~~~~~~~~~~~~~~~~~

Collective分布式任务场景下， ``paddle.distributed.launch`` 组件支持以下参数：

.. code-block::
   usage: launch.py [-h] [--log_dir LOG_DIR] [--nproc_per_node NPROC_PER_NODE]
                    [--run_mode RUN_MODE] [--gpus GPUS] [--ips IPS]
                    training_script ...
   
   启动分布式任务 
   
   optional arguments:
     -h, --help            show this help message and exit
   
   Base Parameters:
     --log_dir LOG_DIR     The path for each process's log. Default
                           --log_dir=log/
     --nproc_per_node NPROC_PER_NODE
                           The number of processes to launch on a node.In gpu
                           training, it should be less or equal to the gpus
                           number of you system(or you set by --gpus). And so
                           each process can bound to one or average number of
                           gpus.
     --run_mode RUN_MODE   run mode of job, can be:collective/ps/ps-heter
     --gpus GPUS           It's for gpu training.For example:--gpus="0,1,2,3"
                           will launch four training processes each bound to one
                           gpu.
     training_script       The full path to the single GPU training
                           program/script to be launched in parallel, followed by
                           all the arguments for the training script
     training_script_args
   
   Collective Parameters:
     --ips IPS             Paddle cluster nodes ips, such as
                           192.168.0.16,192.168.0.17..
   
各个参数的含义如下：

-  log_dir：训练日志储存目录。该目录下包含 ``endpoints.log`` 文件和各个卡的训练日志 ``workerlog.x`` （如workerlog.0，wokerlog.1等），其中 ``endpoints.log`` 文件记录各个训练进程的ip地址和端口号。
-  nproc_per_node： 每个节点上启动进程数，即使用的训练卡数。该进程数需要小于或等于节点上可用卡数。该参数和 ``gpus`` 参数只需要设置其一。
-  run_mode：运行模式，如collecitve，ps（parameter-server）或者ps-heter。
-  gpus：每个节点上使用的gpu卡的列表，以逗号间隔。例如 ``--gpus="0,1,2,3"``\ 。需要注意：这里的指定的卡号为物理卡号，而不是逻辑卡号。
-  training_script：训练脚本，如 ``train.py``\ 。
-  training_script_args：训练脚本的参数。
-  ips：所有训练节点的ip地址列表，以逗号间隔。例如， ``--ips="192.168.0.1,192.168.0.2``\ 。需要注意的是，该列表在所有节点上需要保持一致。
