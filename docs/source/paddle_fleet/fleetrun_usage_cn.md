# fleetrun 启动分布式任务

我们提供`fleetrun`命令，只需一行简单的启动命令，即可轻松地将Paddle Fleet GPU单机单卡任务切换为多机多卡任务，也可将参数服务器单节点任务切换为多个服务节点、多个训练节点的分布式任务。

## 使用要求
使用`fleetrun`命令的要求：
- 安装 paddlepaddle 2.0-rc 及以上

## 使用说明
####  GPU场景
- **GPU单机单卡训练**

多机单卡有两种方式：一种可直接使用`python`执行，也可以使用`fleetrun`执行。**推荐使用`fleetrun`启动方法** 

【方法一】直接使用`python`执行
```sh
 export CUDA_VISIBLE_DEVICES=0
 python train.py
```

【方法二】使用`fleetrun`执行
```
 fleetrun --gpus=0 train.py
```

注：如果指定了`export CUDA_VISIBLE_DEVICES=0` ，则可以直接使用：
```
export CUDA_VISIBLE_DEVICES=0
fleetrun train.py
```


- **GPU单机4卡训练**

```
fleetrun --gpus=0,1,2,3 train.py
```

注：如果指定了```export CUDA_VISIBLE_DEVICES=0,1,2,3``` ，则可以直接使用：
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun train.py
```

- **GPU多机多卡训练**

**[示例一]** 2机8卡 (每个节点4卡)
```sh
fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus=0,1,2,3 train.py
```
注：如果每台机器均指定了```export CUDA_VISIBLE_DEVICES=0,1,2,3``` ，则可以直接在每台节点上启动：
```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" train.py
```

**[示例二]**  2机16卡（每个节点8卡，假设每台机器均有8卡可使用）
```sh
fleetrun --ips="xx.xx.xx.xx,yy.yy.yy.yy" train.py
```

- **GPU 在PaddleCloud上提交任务**

**PaddleCloud**是百度开源的云上任务提交工具，提供云端训练资源，打通⽤户云端资源账号，并且支持以命令行形式进行任务提交、查看、终止等多种功能。PaddleCloud更多详情：[PaddleCloud](https://github.com/PaddlePaddle/PaddleCloud "PaddleCloud")

  在PaddleCloud上启动分布式任务十分方便，无论执行单机单卡还是多机多卡任务，只需使用：
```sh
fleetrun train.py 
```

####  CPU场景

- **参数服务器训练 - 单机训练（0个服务节点，1个训练节点）**

```sh
python train.py
```

- **参数服务器训练 - 单机模拟分布式训练（1个服务节点，4个训练节点）**

```sh
fleetrun --server_num=1 --worker_num=4 train.py
```

- **参数服务器训练 - 多机训练（2台节点，每台节点均有1个服务节点，4个训练节点）**

```sh
 # 2个servers 8个workers
 fleetrun --servers="xx.xx.xx.xx:6170,yy.yy.yy.yy:6171" --workers="xx.xx.xx.xx:6172,xx.xx.xx.xx:6173,xx.xx.xx.xx:6174,xx.xx.xx.xx:6175,yy.yy.yy.yy:6176,yy.yy.yy.yy:6177,yy.yy.yy.yy:6178,yy.yy.yy.yy:6179" train.py
```

- **参数服务器训练 - 在PaddleCloud上提交任务**

由于Paddlecloud对参数服务器训练做了比较完备的封装，因此可以直接使用：
```sh
python train.py
```

## fleetrun参数介绍
- GPU模式相关参数:
	- ips （str，可选）： 指定选择哪些节点IP进行训练，默认为『127.0.0.1』, 即会在本地执行单机单卡或多卡训练。
	- gpus（str, 可选）： 指定选择哪些GPU卡进行训练，默认为None，即会选择`CUDA_VISIBLE_DEVICES`所显示的所有卡。

- 参数服务器模式可配参数:
	- server_num（int，可选）：本地模拟分布式任务中，指定参数服务器服务节点的个数
	- worker_num（int，可选）：本地模拟分布式任务中，指定参数服务器训练节点的个数
	- servers（str, 可选）： 多机分布式任务中，指定参数服务器服务节点的IP和端口
	- workers（str, 可选）： 多机分布式任务中，指定参数服务器训练节点的IP和端口

- 其他：
	- log_dir（str, 可选）： 指定分布式任务训练日志的保存路径，默认保存在"./log/"目录。


## 利用fleetrun将单机单卡任务转换为单机多卡任务
下面我们将通过例子，为您详细介绍如何利用`fleetrun`将单机单卡训练任务转换为单机多卡训练任务。
FleetX提供非常简单易用的代码来实现Imagenet数据集上训练ResNet50模型。
```py
import fleetx as X
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet

configs = X.parse_train_configs()

model = X.applications.Resnet50()
imagenet_downloader = X.utils.ImageNetDownloader()
local_path = imagenet_downloader.download_from_bos(local_path='./data')
local_path = "./data/"
loader = model.load_imagenet_from_file(
    "{}/train.txt".format(local_path), batch_size=32)

fleet.init(is_collective=True)

optimizer = fluid.optimizer.Momentum(
    learning_rate=configs.lr,
    momentum=configs.momentum)
optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(model.loss)

trainer = X.MultiGPUTrainer()
trainer.fit(model, loader, epoch=10)
```
#### 单机单卡训练
将上述代码保存在`res_app.py`代码中，单机单卡训练十分的简单，只需要：
```sh
export CUDA_VISIBLE_DEVICES=0
python res_app.py
```
可以看见终端上打印日志信息：
```sh
--202X-0X-0X 06:42:53--  https://fleet.bj.bcebos.com/models/0.0.4/resnet50_nchw.tar.gz
Connecting to 172.19.57.45:3128... connected.
Proxy request sent, awaiting response... 200 OK
Length: 29733 (29K) [application/x-gzip]
Saving to: ‘/usr/local/lib/python2.7/dist-packages/fleetx/applications/resnet50_nchw.tar.gz’

resnet50_nchw.tar.gz                          100%[==============================================================================================>]  29.04K   127KB/s    in 0.2s

202X-0X-0X 06:42:56 (127 KB/s) - ‘/usr/local/lib/python2.7/dist-packages/fleetx/applications/resnet50_nchw.tar.gz’ saved [29733/29733]
('reader shuffle seed', 0)
('trainerid, trainer_count', 0, 1)
read images from 0, length: 61700, lines length: 61700, total: 61700
worker_index: 0, step11, train_loss: 7.020836, total time cost = 0.286696, step per second: 3.488016, speed: 3.488016
worker_index: 0, step12, train_loss: 6.972931, total time cost = 0.319859, step per second: 6.252759, speed: 30.154240
worker_index: 0, step13, train_loss: 6.851268, total time cost = 0.423936, step per second: 7.076546, speed: 9.608284
worker_index: 0, step14, train_loss: 7.111120, total time cost = 0.527876, step per second: 7.577542, speed: 9.620934
...
```
#### 单机多卡训练
从单机单卡训练到单机多卡训练不需要改动`res_app.py`代码，只需改一行启动命令：
```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun res_app.py
```
训练日志可以在终端上查看，也可稍后在./log/目录下查看每个卡的日志。
终端可以看到显示日志如下：
```sh
-----------  Configuration Arguments -----------
gpus: 0,1,2,3
ips: 127.0.0.1
log_dir: log
server_num: None
servers:
training_script: fleetx_res.py
training_script_args: []
worker_num: None
workers:
------------------------------------------------
INFO 202X-0X-0X 06:09:36,185 launch_utils.py:425] Local start 4 processes. First process distributed environment info (Only For Debug):
=======================================================================================
            Distributed Envs              Value
---------------------------------------------------------------------------------------
PADDLE_CURRENT_ENDPOINT                   127.0.0.1:33360
PADDLE_TRAINERS_NUM                       4
FLAGS_selected_gpus                       0
PADDLE_TRAINER_ENDPOINTS                  ... 0.1:11330,127.0.0.1:54803,127.0.0.1:49294
PADDLE_TRAINER_ID                         0
=======================================================================================
('reader shuffle seed', 0)
('trainerid, trainer_count', 0, 4)
read images from 0, length: 15425, lines length: 15425, total: 61700
worker_index: 0, step11, train_loss: 7.081496, total time cost = 0.113786, step per second: 8.788429, speed: 8.788429
worker_index: 0, step12, train_loss: 7.012076, total time cost = 0.228058, step per second: 8.769704, speed: 8.751059
worker_index: 0, step13, train_loss: 6.998970, total time cost = 0.349108, step per second: 8.593330, speed: 8.261041
.....
```

