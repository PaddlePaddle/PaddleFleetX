
# 快速开始

## 1. 环境准备

这里介绍使用裸机或者 Docker 环境使用 FleetX 的方法，用户根据具体情况选择一种安装部署方式即可。
使用多机训练时，需要在每台机器上都部署相应的环境。

### 1.1 Docker 环境部署

推荐使用 Docker 安装部署 FleetX 进行大模型训练，Docker 环境的安装可以参考[文档](docker_install.md)。

请根据本地 CUDA 版本（使用 `nvidia-smi`命令查看）使用以下命令拉取对应或兼容的镜像，

```
docker pull registry.baidubce.com/kuizhiqing/fleetx-cuda11.2-cudnn8:alpha
```

如本地环境cuda版本较低可以使用以下镜像，并在后续使用中替换。

```
docker pull registry.baidubce.com/kuizhiqing/fleetx-cuda10.2-cudnn7:alpha
```

大模型训练需要使用GPU，如已安装 nvida-container-runtime 可以使用以下命令运行镜像，

```
docker run -it --name=paddle --net=host -v /dev/shm:/dev/shm --shm-size=32G -v $PWD:/paddle --runtime=nvidia registry.baidubce.com/kuizhiqing/fleetx-cuda11.2-cudnn8:alpha bash
```

未安装 nvida-container-runtime 或启动后无法执行 `nvidia-smi` 查看GPU信息时可以尝试通过如下脚本启动运行，

```shell
export CUDA_SO="$(\ls /usr/lib64/libcuda* | grep -v : | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | grep -v : | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(find /dev/nvidia* -maxdepth 1 -not -type d | xargs -I{} echo '--device {}:{}')

nvsmi=`which nvidia-smi`

docker run \
${CUDA_SO} ${DEVICES} \
-v /dev/shm:/dev/shm \
-v $PWD:/paddle \
--name paddle \
--net=host \
--shm-size=32G \
-v $nvsmi:$nvsmi \
-it \
registry.baidubce.com/kuizhiqing/fleetx-cuda11.2-cudnn8:alpha \
bash
```

以上命令 `-v $PWD:/paddle` 将当前目录映射到 /paddle 目录，在 docker 环境内部对该目录的更改将会持久化。

> 为保证通信效率和通信正常，添加参数 --net=host 使用主机网络，更多 docker run 参数说明请参考 [docker 文档](https://docs.docker.com/engine/reference/commandline/run/)。

### 1.2 裸机部署

**安装 PaddlePaddle**

首先根据环境在
[安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html) 选择对应的版本使用 pip install 执行对应命令安装 PaddlePaddle.
**请务必按照文档安装 GPU 版本且验证安装成功**。

例如使用如下命令将会安装基于 CUDA 11.2 最新版本的 PaddlePaddle. 

```shell
python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

安装遇到问题以及环境验证的方法也可以参考[文档](deployment_faq.md#1-单机环境验证)。

**安装依赖**

使用以下命令安装 FleetX 运行所需依赖。

```shell
python -m pip install -r https://raw.githubusercontent.com/PaddlePaddle/FleetX/develop/requirements.txt -i https://mirror.baidu.com/pypi/simple
```

## 2. 单机多卡训练

进入环境后首先使用以下命令拉取最新代码

```
git clone https://github.com/PaddlePaddle/FleetX.git
```

切换工作目录并下载demo数据，
```
cd FleetX/examples/gpt/hybrid_parallel # 或者 cd FleetX/examples/gpt/auto_parallel

mkdir data
wget -O data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget -O data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

然后使用以下命令运行单机多卡程序，

```
python -m paddle.distributed.launch run_pretrain.py -c ./configs_1.3B_dp8.yaml
```

若要在显存容量更小的环境例如 16G 显存下进行GPT模型单机训练，可通过减小`Model.hidden_size`调整模型规模至合适大小再启动训练，命令如下：

```
python -m paddle.distributed.launch run_pretrain.py -c ./configs_1.3B_dp8.yaml -o Model.hidden_size=1024
```

> 更多 launch 启动参数和用法请参考 [API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/launch_cn.html)。

成功则开始训练过程，
```
LAUNCH INFO 2022-08-15 07:37:38,946 -----------  Configuration  ----------------------
LAUNCH INFO 2022-08-15 07:37:38,946 devices: None
LAUNCH INFO 2022-08-15 07:37:38,947 elastic_level: -1
LAUNCH INFO 2022-08-15 07:37:38,947 elastic_timeout: 30
LAUNCH INFO 2022-08-15 07:37:38,947 gloo_port: 6767
LAUNCH INFO 2022-08-15 07:37:38,947 host: None
LAUNCH INFO 2022-08-15 07:37:38,947 ips: None
LAUNCH INFO 2022-08-15 07:37:38,947 job_id: default
LAUNCH INFO 2022-08-15 07:37:38,947 legacy: False
LAUNCH INFO 2022-08-15 07:37:38,947 log_dir: log
LAUNCH INFO 2022-08-15 07:37:38,947 log_level: INFO
LAUNCH INFO 2022-08-15 07:37:38,947 master: None
LAUNCH INFO 2022-08-15 07:37:38,947 max_restart: 3
LAUNCH INFO 2022-08-15 07:37:38,947 nnodes: 1
LAUNCH INFO 2022-08-15 07:37:38,947 nproc_per_node: None
LAUNCH INFO 2022-08-15 07:37:38,947 rank: -1
LAUNCH INFO 2022-08-15 07:37:38,947 run_mode: collective
LAUNCH INFO 2022-08-15 07:37:38,947 server_num: None
LAUNCH INFO 2022-08-15 07:37:38,947 servers:
LAUNCH INFO 2022-08-15 07:37:38,947 start_port: 6070
LAUNCH INFO 2022-08-15 07:37:38,947 trainer_num: None
LAUNCH INFO 2022-08-15 07:37:38,947 trainers:
LAUNCH INFO 2022-08-15 07:37:38,947 training_script: run_pretrain.py
LAUNCH INFO 2022-08-15 07:37:38,947 training_script_args: ['-c', './configs_1.3B_dp8.yaml']
LAUNCH INFO 2022-08-15 07:37:38,947 with_gloo: 1
LAUNCH INFO 2022-08-15 07:37:38,947 --------------------------------------------------
LAUNCH INFO 2022-08-15 07:37:38,948 Job: default, mode collective, replicas 1[1:1], elastic False
LAUNCH INFO 2022-08-15 07:37:38,949 Run Pod: vqhbut, replicas 8, status ready
LAUNCH INFO 2022-08-15 07:37:39,063 Watching Pod: vqhbut, replicas 8, status running
## 启动配置
[2022-08-15 08:36:12,563] [    INFO] - global step 1, epoch: 0, batch: 0, loss: 11.255846024, avg_reader_cost: 0.10940 sec, avg_batch_cost: 4.78177 sec, speed: 0.21 step/s, ips_total: 13705 tokens/s, ips: 1713 tokens/s, learning rate: 5.55556e-09
## 更多训练日志
```

如有启动异常请根据[文档](deployment_faq.md#1-单机环境验证)进行工作环境验证，其他问题可参考[FAQ](deployment_faq.md#3-faq)解决。

## 3. 多机多卡训练

使用以下命令进行多机分布式训练，其中 --nnodes 参数为分布式训练机器数量，--master 为训练机器中其中一台机器的IP，运行时需要将命令中示例IP替换为真实的机器IP和任意可用端口，然后在**每个节点**上都运行以下命令，
如果不知道机器IP可以不设置--master参数先在一台机器上启动，然后根据提示复制命令在其他机器上启动即可。

```
python -m paddle.distributed.launch --master=10.10.10.1:8099 --nnodes=2 run_pretrain.py -c ./configs_6.7B_sharding16.yaml
```

> 该示例为16卡任务，需要满足总卡数为16的要求。

> 注意这里需要使用单机多卡训练部分的代码和数据。


成功则开始多机训练过程，日志和单机多卡类似，日志异常时请按照[文档](deployment_faq.md#2-分布式环境验证)进行环境验证和问题排查。

若要在显存容量更小的环境例如 16G 显存下进行GPT模型单机训练，可通过减小`Model.hidden_size`调整模型规模至合适大小再启动训练，命令如下：

```
python -m paddle.distributed.launch --master=10.10.10.1:8099 --nnodes=2 run_pretrain.py -c ./configs_6.7B_sharding16.yaml -o Model.hidden_size=2048
```

更多大模型多机训练内容可见[文档](../examples/gpt/hybrid_parallel/README.md)。
