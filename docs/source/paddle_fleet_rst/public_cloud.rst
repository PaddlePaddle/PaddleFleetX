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
'使用自定义镜像创建训练作业 <https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0087.html>'

制作paddle docker镜像
^^^^^

-  准备dockerfile

.. code-block::
    # modelarts提供了各种类型的基础镜像，详细：https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0217.html#modelarts_23_0217__section1126616610513，请根据需要按需选择基础镜像，该示例中选择的是cpu镜像
    FROM swr.cn-north-4.myhuaweicloud.com/modelarts-job-dev-image/custom-cpu-base:1.3

    # 安装Paddle，详细：https://www.paddlepaddle.org.cn/，该示例选择的是Paddle 2.0.1\Ubuntu\pip\CPU版本
    RUN python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

-  构建docker镜像

.. code-block::

    docker build -f Dockerfile . -t swr.cn-north-4.myhuaweicloud.com/deep-learning-diy/paddle-cpu-2.0.1:latest

-  将docker镜像推入镜像仓库

.. code-block::

    docker push swr.cn-north-4.myhuaweicloud.com/deep-learning-diy/paddle-cpu-2.0.1:latest

准备运行脚本(Collective模式)
^^^^^

-  运行脚本

run.sh

.. code-block::

if [[ $NUM == 1 ]]; then
    config="--selected_gpus=0,1,2,3,4,5,6,7 --log_dir mylog"
    python -m paddle.distributed.launch ${config} train.py
else
    python -m paddle.distributed.launch \
        --cluster_node_ips=192.168.1.2,192.168.1.3 \
        --node_ip=192.168.1.3 \
        --started_port=6170 \
        --selected_gpus=0,1,2,3 \
        train_with_fleet.py
fi

-  组网代码

train_with_fleet.py

.. code-block::
# -*- coding: utf-8 -*-
import os
import numpy as np
import paddle.fluid as fluid
# 区别1: 导入分布式训练库
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.base import role_maker

# 定义网络
def mlp(input_x, input_y, hid_dim=1280, label_dim=2):
    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim, act='tanh')
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim, act='tanh')
    prediction = fluid.layers.fc(input=[fc_2], size=label_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
    avg_cost = fluid.layers.mean(x=cost)
    return avg_cost 
    
# 生成数据集
def gen_data():
    return {"x": np.random.random(size=(128, 32)).astype('float32'),
            "y": np.random.randint(2, size=(128, 1)).astype('int64')}

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')

# 定义损失 
cost = mlp(input_x, input_y)
optimizer = fluid.optimizer.SGD(learning_rate=0.01)

# 区别2: 定义训练策略和集群环境定义
dist_strategy = DistributedStrategy()
role = role_maker.PaddleCloudRoleMaker(is_collective=True)
fleet.init(role)

# 区别3: 对optimizer封装，并调用封装后的minimize方法
optimizer = fleet.distributed_optimizer(optimizer, strategy=DistributedStrategy())
optimizer.minimize(cost, fluid.default_startup_program())

train_prog = fleet.main_program


# 获得当前gpu的id号
gpu_id = int(os.getenv("FLAGS_selected_gpus", "0"))
print(gpu_id)
place = fluid.CUDAPlace(gpu_id)

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

step = 100
for i in range(step):
    cost_val = exe.run(program=train_prog, feed=gen_data(), fetch_list=[cost.name])
    print("step%d cost=%f" % (i, cost_val[0]))

# 区别4: 模型保存
model_path = "./"
if os.path.exists(model_path):
    fleet.save_persistables(exe, model_path)

提交分布式训练任务
^^^^^




注意：如果是GPU或者Ascend（NPU），ModelArts会根据当前节点的GPU/Ascend（NPU）数量来自动启动多进程，

2.3、阿里云PAI平台
^^^^^^^^

由于阿里云PAI平台不支持自定义框架的方式来提交训练任务，目前 paddle 还无法在阿里云PAI平台上运行。