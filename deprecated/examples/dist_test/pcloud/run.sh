##                       类型作业演示                        ##
## 请将下面的 user_ak/user_sk 替换成自己的 ak/sk             ##
## 请将下面的 cluster_name 替换成所在组关联的k8s集群名称     ##
##                                                           ##
###############################################################
# 请更新成所在组下的个人 access key/secret key
AK=""
SK=""
# 作业参数
gpus_per_node="8"
k8s_gpu_type="baidu/gpu_v100"
k8s_wall_time="99:00:00"
k8s_priority="high"
is_standalone="0"
k8s_trainers="1"
JOB_NAME="mnist_n${k8s_trainers}"
# 请替换成所在组关联的集群名称
cluster_name="v100-32-0-cluster"
group_name="dltp-0-yq01-k8s-gpu-v100-8"
# 作业版本
job_version="paddle-fluid-v1.8.2"

job_name=${JOB_NAME}
# 线上正式环境
server="paddlecloud.baidu-int.com"
port=80

distributed_conf="1 "
if [ ${k8s_trainers} -gt 1 ]
then
    distributed_conf="0 --distribute-job-type NCCL2 "
fi

upload_files="before_hook.sh end_hook.sh ../*.py"

# 启动命令
start_cmd="python -m paddle.distributed.launch \
                  --selected_gpus=0,1 \
                  --log_dir=mylog \
                  train.py --distributed"

paddlecloud job train \
    --job-name ${job_name} \
    --group-name ${group_name} \
    --cluster-name ${cluster_name} \
    --job-conf job.cfg \
    --start-cmd "${start_cmd}" \
    --files ${upload_files} \
    --job-version ${job_version}  \
    --k8s-gpu-cards $gpus_per_node \
    --k8s-wall-time ${k8s_wall_time} \
    --k8s-cpu-cores 35 \
    --k8s-trainers ${k8s_trainers} \
    --k8s-priority ${k8s_priority} \
    --is-standalone ${distributed_conf} 
