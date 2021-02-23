#!/bin/bash
date && echo 'begin build env...'

PADDLEBOX_HOME=`pwd`/heterbox

function fatal_error() {
   echo -e "FATAL: " "$1" >> /dev/stderr
   exit 1
}

function download_python() {
    wget ftp://yq01-ps-7-m12-wise056.yq01.baidu.com/home/work/wxx/fengdanlei/upload/python.tar.gz &> /dev/null
    tar -zvxf python.tar.gz &> /dev/null
    rm python.tar.gz
}

function install_heterbox() {
    export PATH=`pwd`/python/bin:$PATH
    export LD_LIBRARY_PATH=`pwd`/python/lib:$LD_LIBRARY_PATH
    `pwd`/python/bin/python -m pip install bin/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl -U
}

function download_cuda() {
    wget ftp://yq01-ps-7-m12-wise056.yq01.baidu.com/home/work/wxx/date/2020/1228/cuda-10.2.tar.gz &> /dev/null
    [ $? -ne 0 ] && fatal_error "wget cuda-10.2.tar.gz error..."
    tar zxvf cuda-10.2.tar.gz &> /dev/null
    [ $? -ne 0 ] && fatal_error "unzip cuda-10.2.tar.gz error..."
    rm -f cuda-10.2.tar.gz
}

function download_cudnn() {
    wget ftp://yq01-ps-7-m12-wise056.yq01.baidu.com/home/work/wxx/date/2020/1228/cudnn_v7.4.tgz
    mkdir ./cudnn_v7.4/
    tar -zvxf cudnn_v7.4.tgz -C ./cudnn_v7.4/ &> /dev/null
}

function download_nccl() {
    wget data-im.baidu.com:/home/work/var/CI_DATA/im/static/nccl2.7.3_cuda10.2.tar.gz/nccl2.7.3_cuda10.2.tar.gz.1 -O nccl2.7.3_cuda10.2.tar.gz &> /dev/null
    [ $? -ne 0 ] && fatal_error "wget nccl2.7.3_cuda10.2.tar.gz error..."
    tar zxvf nccl2.7.3_cuda10.2.tar.gz &> /dev/null
    [ $? -ne 0 ] && fatal_error "unzip nccl2.7.3_cuda10.2.tar.gz error..."
    rm -f nccl2.7.3_cuda10.2.tar.gz
}

unset http_proxy
unset https_proxy
download_python
[ $? -ne 0 ] && fatal_error "download python failed"
install_heterbox
[ $? -ne 0 ] && fatal_error "install heterbox failed"
download_cuda
[ $? -ne 0 ] && fatal_error "download cuda failed"
download_cudnn
[ $? -ne 0 ] && fatal_error "download cudnn failed"
download_nccl
[ $? -ne 0 ] && fatal_error "download nccl failed"
date && echo 'end build env...'
exit 0
