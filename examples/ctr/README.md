# Benchmark for CTR
Benchmark for https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr

# To get dataset, you can run:
```
sh get_data.sh
```

# To get training throughputs
```
python -m distributed.launch_ps --worker_num 2 --server_num 2 dist_ctr.py
```
You will get 55000 instances/s
