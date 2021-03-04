
ARG PADDLE_IMG=registry.baidubce.com/paddlepaddle/paddle:2.0.0

FROM ${PADDLE_IMG}

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list

RUN apt update && \
    apt install -y procps curl wget git vim

## 根据需要安装
RUN python3 -m pip --no-cache-dir install -i https://mirror.baidu.com/pypi/simple --no-cache-dir \
    hello-world
