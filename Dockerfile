ARG BASE_IMAGE=registry.baidubce.com/paddlepaddle/paddle:2.4.1-gpu-cuda11.2-cudnn8.2-trt8.0

FROM $BASE_IMAGE

WORKDIR /paddle

RUN python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# RUN wget https://raw.githubusercontent.com/PaddlePaddle/PaddleFleetx/develop/requirements.txt && python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
COPY requirements.txt /paddle

RUN python -m pip install -r requirements.txt #-i https://mirror.baidu.com/pypi/simple

ENV LD_LIBRARY_PATH=/usr/lib64/:${LD_LIBRARY_PATH}

