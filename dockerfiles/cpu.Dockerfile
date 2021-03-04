
ARG PADDLE_IMG=registry.baidubce.com/paddlepaddle/paddle:2.0.0

FROM ${PADDLE_IMG}

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list

RUN apt update && \
    apt install -y procps curl wget git vim

