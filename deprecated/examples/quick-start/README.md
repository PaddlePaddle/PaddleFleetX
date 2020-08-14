
# Quick start examples for Fleet API

## Collective Training

```
python -m paddle.distributed.launch collective_train.py
```

## Parameter Server Training

```
python -m paddle.distributed.launch_ps --worker_num 2 --server_num 2 distributed_train.py
```

