# To get dataset, you can run:
```
sh get_data.sh
```

# To simulation distributed training with parameter server
```
python -m paddle.distributed.launch_ps --worker_num 2 --server_num 2 train.py
```

# To run local training
```
python train.py --is_local 1
```
