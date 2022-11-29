# GPT 自动并行模型训练

分布式并行训练技术使超大模型成为可能，但分布式训练程序的编写门槛较高，并行算法较为复杂，开发者需同时具有较好的工程能力和算法功底。为了降低分布式训练的难度，自动并行成为新的研究热点，受到学术界和工业界的广泛关注。自动并行通常分为半自动并行和全自动并行。半自动并行指的是开发者在单机脚本的基础上额外添加少量标注信息即可表达并行逻辑。而全自动并行则无需开发者添加任何并行逻辑，根据单机脚本自动搜索出较为高效的并行策略，实现分布式训练。

## 运行方式
本目录按照345M、1.3B和6.7B规模大小，给出32G V100环境下GPT模型半自动并行训练的策略配置如下：

| 模型规模   | 训练策略                     | yaml文件                               |
|----------|---------------------------- |----------------------------------------|
| 345MB    | 单卡+fp16                    | pretrain_gpt_345M_single_card.yaml     |
| 1.3B     | dp8+fp16+recompute          | pretrain_gpt_1.3B_dp8.yaml             |
| 6.7B     | sharding16+fp16+recompute   | pretrain_gpt_6.7B_sharding16.yaml  |

若要在显存容量更小的16G V100环境下进行GPT大模型训练，可将对应yaml文件中的`Model`-`hidden size`值改为原来的1/2即可。

### 策略支持

自动并行包括2种模式：半自动并行与全自动并行。
半自动并行包括了数据并行、张量模型并行、流水线并行和分组切片并行。此外还支持重计算、混合精度等策略，来减少显存占用、加速训练。**目前，GPT 模型训练可以支持任意维度的策略组合。**

|                 | data parallel | tensor parallel | pipeline parallel | pure fp16 | recompute |
|-----------------|---------------|-----------------|-------------------|-----------|-----------|
| sharding stage1 | ✓             | ✓               | ✓                 | ✓         | ✓         |
| sharding stage2 | ✓             | ✓               | ✓                 | ✓         | ✓         |
| sharding stage3 | ✓             | ✓               | ✓                 | ✓         | ✓         |


### 单卡训练

以单机1.3B模型训练为例，该gpt程序需要单卡32G V100以运行

**启动命令**
```shell
cd PaddleFleetX # 如果已在 PaddleFleetX 根目录下，则忽略

export FLAGS_USE_STANDALONE_EXECUTOR=False # 设置执行器环境变量
python ./tools/auto.py -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_single_card.yaml
```

### 单机训练

以单机1.3B模型数据并行训练为例，通过``paddle.distributed.launch``启动多进程训练，该gpt程序需要8卡32G V100以运行。

**启动命令**
```shell
cd PaddleFleetX # 如果已在 PaddleFleetX 根目录下，则忽略

log_dir=log_auto
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    ./tools/auto.py \
    -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp8.yaml
```

若要在显存容量更小的16G V100环境下进行GPT模型单机训练，可通过减小`Model.hidden_size`调整模型规模至合适大小再启动训练，命令如下：

**启动命令**
```shell
log_dir=log_auto
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    ./tools/auto.py \
    -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp8.yaml \
    -o Model.hidden_size=1024
```

每张GPU的运行日志`workerlog.x`可在launch命令中指定的`log_dir`路径下找到；若未指定，日志路径为`log/workerlog.x`。运行日志具体内容如下：

**运行日志**

```
[INFO 2022-08-19 10:47:00,392 engine.py:461] [train] epoch: 0 step: 0 lr: 5.555556e-09 loss: 10.972320
[INFO 2022-08-19 10:47:02,858 engine.py:461] [train] epoch: 0 step: 1 lr: 8.333333e-09 loss: 10.950481
[INFO 2022-08-19 10:47:05,321 engine.py:461] [train] epoch: 0 step: 2 lr: 1.111111e-08 loss: 10.951584
[INFO 2022-08-19 10:47:07,791 engine.py:461] [train] epoch: 0 step: 3 lr: 1.388889e-08 loss: 10.954518
[INFO 2022-08-19 10:47:10,256 engine.py:461] [train] epoch: 0 step: 4 lr: 1.666667e-08 loss: 10.959060
[INFO 2022-08-19 10:47:12,725 engine.py:461] [train] epoch: 0 step: 5 lr: 1.944444e-08 loss: 10.957585
[INFO 2022-08-19 10:47:15,198 engine.py:461] [train] epoch: 0 step: 6 lr: 2.222222e-08 loss: 10.947868
[INFO 2022-08-19 10:47:17,680 engine.py:461] [train] epoch: 0 step: 7 lr: 2.500000e-08 loss: 10.939037
```

### 多机训练

若需要在更多机器上进行大模型训练，则需要在每个参与训练的节点上设置master节点ip/port信息后执行启动命令（master节点ip为训练所用某一台机器的ip即可）。

以2机16卡32G V100上的6.7B模型分组切分并行训练为例，启动命令为：

```shell
master_ip=master节点ip
master_port=可用的空闲端口号

log_dir=log_sharding16
python -m paddle.distributed.launch --log_dir $log_dir \
    --master=$master_ip:$master_port --nnodes=2 --devices "0,1,2,3,4,5,6,7" \
    ./tools/auto.py -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_6.7B_sharding16.yaml
```

若要在显存容量更小的16G V100环境下进行GPT模型两机训练，也可通过减小`Model.hidden_size`调整模型规模至合适大小再启动训练，命令如下：

```shell
master_ip=master节点ip
master_port=可用的空闲端口号

log_dir=log_sharding16
python -m paddle.distributed.launch --log_dir $log_dir \
    --master=$master_ip:$master_port --nnodes=2 --devices "0,1,2,3,4,5,6,7" \
    ./tools/auto.py -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_6.7B_sharding16.yaml \
    -o Model.hidden_size=2048
```
