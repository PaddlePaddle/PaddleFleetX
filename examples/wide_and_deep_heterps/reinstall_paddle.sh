#!/bin/bash
# Usage : sh reinstall_paddle.sh
# 1. put your new paddlpaddle whl package into ./bin/
# 2. use /home/work/python environment in docker by default
# 3. run reinstall_paddle.sh

date && echo 'begin install paddlepaddle whl...'

function fatal_error() {
   echo -e "FATAL: " "$1" >> /dev/stderr
   exit 1
}

function install_heterps() {
    export PATH=/home/work/python/bin:$PATH
    export LD_LIBRARY_PATH=/home/work/python/lib:$LD_LIBRARY_PATH
    /home/work/python/bin/python -m pip install bin/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl -U
}

unset http_proxy
unset https_proxy
install_heterps
[ $? -ne 0 ] && fatal_error "install paddlepaddle whl failed"
date && echo 'end install paddlepaddle whl...'
exit 0
