# Suppose there are two nodes available for training, the ip of node 1
# is 192.168.0.1 and that of node 2 is 192.168.0.2.
# this script shows the start command on node 2

current_node_ip=192.168.0.2
cluster_node_ips=192.168.0.1,192.168.0.2

CUDA_VISIBLE_DEVICES=0,1 \
  python -m paddle.distributed.launch \
  --selected_gpus=0,1 \
  --cluster_node_ips=$cluster_node_ips \
  --node_ip=$current_node_ip \
  train.py --distributed
