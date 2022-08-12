
# 快速开始

## 拉取镜像

```
docker pull registry.baidubce.com/kuizhiqing/fleetx-cuda11.2-cudnn8:alpha
```

出现问题

* 安装 docker link


## 运行镜像

```
docker run -it --rm --net=host -v /dev/shm:/dev/shm --shm-size=32G -v $PWD:/paddle --runtime=nvidia registry.baidubce.com/kuizhiqing/fleetx-cuda11.2-cudnn8:alpha bash
```

* 安装 docker runtime link


## 单机多卡

```
cd /paddle/examples/gpt/hybrid_parallel
bash run.sh
```

## 多机机多卡

```
cd /paddle/examples/gpt/hybrid_parallel
bash run.sh 2
```

* 网络问题 link

