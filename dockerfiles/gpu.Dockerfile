
ARG CUDA=10.2-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:${CUDA}

ENV LANG C.UTF-8

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    "pip<20.3" \
    setuptools

RUN ln -s $(which python3) /usr/local/bin/python

ARG PADDLE_PKG=paddlepaddle-gpu
ARG PADDLE_VER=2.0.0
RUN python3 -m pip --no-cache-dir install -i https://mirror.baidu.com/pypi/simple --no-cache-dir ${PADDLE_PKG}${PADDLE_VER:+==${PADDLE_VER}}

RUN apt install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx bison graphviz libjpeg-dev zlib1g-dev

