启动分布式任务
------------------

飞桨通过\ ``paddle.distributed.launch``\ 组件启动分布式任务。该组件可用于启动单机多卡分布式任务，也可以用于启动多机多卡分布式任务。该组件为每张参与分布式任务的训练卡启动一个训练进程。默认情形下，该组件将在每个节点上启动\ ``N``\ 个进程，这里\ ``N``\ 等于训练节点的卡数，即使用所有的训练卡。用户也可以通过\ ``gpus``\ 参数指定训练节点上使用的训练卡列表，该列表以逗号分隔。需要注意的是，所有节点需要使用相同数量的训练卡数。

为了启动多机分布式任务，需要通过\ ``ips``\ 参数指定所有节点的IP地址列表，该列表以逗号分隔。需要注意的是，该列表在所有节点上需要保持一致，即各节点IP地址出现的顺序需要保持一致。

例如，可以通过下面的命令启动单机多卡分布式任务，假设节点包含8张GPU卡：

.. code-block::

   python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 train.py --batch_size=64

其中，\ ``train.py``\ 为用户训练脚本，后面可进一步增加脚本参数，如batch size等。用户也可以只使用部分卡进行训练。例如，下面的例子中，仅使用2、3两张卡进行训练：

.. code-block::

   python -m paddle.distributed.launch --gpus 2,3 train.py --batch_size=64

从单机到多机分布式任务，只需额外指定\ ``ips``\ 参数即可，其内容为多机的IP列表。假设两台机器的IP地址分别为192.168.0.1和192.168.0.2，那么在这两个节点上启动多机分布式任务的命令如下所示：

.. code-block::
   
   # 第一个节点
   python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64

   # 第二个节点
   python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64

需要注意的是，两个节点上\ ``ips``\ 列表的顺序需要保持一致。用户也可使用\ ``gpus``\ 参数指定每个节点上只使用部分训练卡，命令如下所示：

.. code-block::
   
   # 第一个节点
   python -m paddle.distributed.launch --gpus=0,1,2,3 --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64
   # 第二个节点
   python -m paddle.distributed.launch --gpus=0,1,2,3 --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64

两个节点上可以使用不同的训练卡进行训练，但需要使用相同数量的训练卡。例如，第一个节点使用0、1两张卡，第二个节点使用2、3两张卡，启动命令如下所示：

.. code-block::
   
   # 第一个节点
   python -m paddle.distributed.launch --gpus=0,1 --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64
   # 第二个节点
   python -m paddle.distributed.launch --gpus=2,3 --ips="192.168.0.1,192.168.0.2" train.py --batch_size=64

下面将介绍\ ``paddle.distributed.launch``\ 组件在不同场景下的详细使用方法。

Collective分布式任务
~~~~~~~~~~~~~~~~~~~~~

Collective分布式任务场景下，\ ``paddle.distributed.launch``\ 组件支持以下参数：

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

-  log_dir：训练日志储存目录。该目录下包含\ ``endpoints.log``\ 文件和各个卡的训练日志 \ ``workerlog.x``\ （如workerlog.0，wokerlog.1等），其中\ ``endpoints.log``\ 文件记录各个训练进程的IP地址和端口号。
-  run_mode：运行模式，如collecitve，ps（parameter-server）或者ps-heter。
-  gpus：每个节点上使用的gpu卡的列表，以逗号间隔。例如\ ``--gpus="0,1,2,3"``\ 。需要注意：这里的指定的卡号为物理卡号，而不是逻辑卡号。
-  training_script：训练脚本，如\ ``train.py``\ 。
-  training_script_args：训练脚本的参数，如batch size和学习率等。
-  ips：所有训练节点的IP地址列表，以逗号间隔。例如，\ ``--ips="192.168.0.1,192.168.0.2``\ 。需要注意的是，该列表在所有节点上需要保持一致，即各节点IP地址出现的顺序在所有节点的任务脚本中需要保持一致。

通过\ ``paddle.distributed.launch``\ 组件启动分布式任务，将在控制台显示第一张训练卡对应的日志信息，并将所有的日志信息保存到\ ``log_dir``\ 参数指定的目录中；每张训练卡的日志对应一个日志文件，形式如\ ``workerlog.x``\ 。

PaddleCloud平台
===================

当在百度内部PaddleCloud平台使用飞桨分布式时，可以省略\ ``ips``\ 参数。假设使用两台机器执行分布式任务，则命令行如下所示：

.. code-block::
   
   # 第一台机器：
   python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" train.py

   # 第二台机器：
   python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" train.py

更多关于如何通过在PaddleCloud上启动分布式任务，请参考PaddleCloud官方文档。

物理机或docker环境启动分布式任务
============================

我们以下图为例说明如何在物理机环境或者docker环境中启动飞桨分布式任务。我们有两台机器，每台机器包含4张GPU卡。两台机器的IP地址分别为192.168.0.1和192.168.0.2。该IP地址可以为两台物理机的IP地址，也可以为两台机器内部Docker容器的IP地址。

.. image:: ../collective/img/dp_exam1.png
  :width: 400
  :alt: launch
  :align: center

为了在两台机器上启动分布式任务，首先需要确保两台机器间的网络是互通的。可以通过\ ``ping`` \命令验证两台机器间的网络互通性，如下所示：

.. code-block::
   
   # 第一个节点
   ping 192.168.0.2
   # 第二个节点
   ping 192.168.0.1

如果两台机器间的网络无法连通，请联系您的网络管理员获取帮助。

假设用户的训练脚本为\ ``train.py``\ ，则可以通过如下命令在两台机器上启动分布式训练任务： 

.. code-block::
   
   # 第一台机器：192.168.0.1
   python -m paddle.distributed.launch --gpus="0,1,2,3" --ips="192.168.0.1,192.168.0.2" train.py

   # 第二台机器：192.168.0.2
   python -m paddle.distributed.launch --gpus="0,1,2,3" --ips="192.168.0.1,192.168.0.2" train.py

当每台机器均使用所有4张训练卡时，也可以省略\ ``gpus``\ 参数，如下所示：

.. code-block::
   
   # 第一台机器：192.168.0.1
   python -m paddle.distributed.launch --ips="192.168.0.1,192.168.0.2" train.py

   # 第二台机器：192.168.0.2
   python -m paddle.distributed.launch --ips="192.168.0.1,192.168.0.2" train.py

用户也可以通过\ ``gpus``\ 参数指定只使用部分训练卡，例如只使用0、1两张卡：

.. code-block::
   
   # 第一台机器：192.168.0.1
   python -m paddle.distributed.launch --gpus="0,1" --ips="192.168.0.1,192.168.0.2" train.py

   # 第二台机器：192.168.0.2
   python -m paddle.distributed.launch --gpus="0,1" --ips="192.168.0.1,192.168.0.2" train.py

通过\ ``paddle.distributed.launch``\ 组件启动分布式任务时，该组件将为\ ``gpus``\ 参数指定的每张训练卡启动一个训练进程。为了实现进程间通信，该组件同时为每个进程绑定一个端口号，进程的IP地址和端口号成为该进程的网络地址。\ ``paddle.distributed.launch``\ 组件随机查找机器上的可用端口，作为训练进程的端口号。假设，Node 0上4个训练进程的端口号分别为3128、5762、6213和6170，则该机器上4个训练进程的网络地址分别为: \ ``192.168.0.1:3128``\ 、\ ``192.168.0.1:5762``\ 、\ ``192.168.0.1:6213``\ 和\ ``192.168.0.1:6170``\ 。当\ ``paddle.distributed.launch``\ 组件无法获取足够的可用端口时，任务启动失败。