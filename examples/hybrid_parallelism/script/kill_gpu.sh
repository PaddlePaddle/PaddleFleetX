lsof /dev/nvidia* |  awk '{print $2}' | xargs -I {} kill -9 {}
