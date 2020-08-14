python -m paddle.distributed.launch \
    --cluster_node_ips=127.0.0.1 \
	--node_ip=127.0.0.1 \
	--selected_gpus="0,1,2,3,4,5,6,7" \
	--log_dir=mylog \
    collective_train.py
