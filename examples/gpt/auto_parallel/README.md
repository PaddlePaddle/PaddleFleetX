# GPT 自动并行模型训练

分布式并行训练技术使超大模型成为可能，但分布式训练程序的编写门槛较高，并行算法较为复杂，开发者需同时具有较好的工程能力和算法功底。为了降低分布式训练的难度，自动并行成为新的研究热点，受到学术界和工业界的广泛关注。自动并行通常分为半自动并行和全自动并行。半自动并行指的是开发者在单机脚本的基础上额外添加少量标注信息即可表达并行逻辑。而全自动并行则无需开发者添加任何并行逻辑，根据单机脚本自动搜索出较为高效的并行策略，实现分布式训练。


## 参数释义


### 数据集
数据集参数指定训练的batch size，以及数据的目录。

```yaml
Data:
  batch_size:
    global_batch_size: 8

  dataset:
    input_dir: ./data
    split: '949,50,1'
    max_seq_len: 1024
```

其中参数对应的释义如下：
| **参数名**                      | **参数释义**               |
|------------------------------|------------------------|
| global_batch_size | 全局的batch size大小，即一次参数更新等效的batch size |
| input_dir         | 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件       |
| split             | 训练集，验证集和测试集的切分比例                     |
| max_seq_len       | 输入文本序列的长度                            |


### 模型网络

网络部分完成了网络的组网操作和自动并行半自动策略的适配，GPT在[FleetX/fleetx/models/gpt_model/modeling_auto.py]([../../ppocr/modeling](https://github.com/PaddlePaddle/FleetX/tree/develop/fleetx/models/gpt_model))下。 
可以使用配置文件配置模型的规模，如：

```yaml
  Model:
    vocab_size: 50304
    hidden_size: 1024
    num_layers: 24
    num_attention_heads: 16
    ffn_hidden_size:
    hidden_dropout_prob: 0.1
    attention_probs_dropout_prob: 0.1
    max_position_embeddings: 1024
    type_vocab_size: 16
    initializer_range: 0.02
```

其中参数对应的释义如下：
| **参数名**                      | **参数释义**               |
|------------------------------|------------------------|
| vocab_size                   | 训练词表大小                 |
| hidden_size                  | 隐藏层大小                  |
| num_layers                   | transformer层数          |
| num_attention_heads          | attention head的数量      |
| max_seq_len                  | 输入文本序列的长度              |
| ffn_hidden_size              | ffn层大小，一般为隐藏层的四倍       |
| attention_probs_dropout_prob | attention中的dropout的失活率 |
| max_position_embeddings      | position embedding的长度  |
| type_vocab_size              | 词表类型                   |
| initializer_range            | 参数初始化的范围               |


### 优化器

GPT训练默认使用AdamW优化器以及cosine学习率衰减，这里通过配置文件配置优化器的参数，如：

```yaml
  Optimizer:
    # name: Adam
    weight_decay: 0.01
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_epsilon: 1.0e-8
    lr:
      # name: consine
      decay_steps: 360000
      # max_steps: 500000
      warmup_rate: 0.01
      max_lr: 1.0e-5
      min_lr: 5.0e-5
    grad_clip: 1.0
```

其中参数说明：

| **参数名**      | **参数释义**                  |
|--------------|---------------------------|
| weight_decay | weight的衰减率                |
| adam_beta1   | 一阶矩估计的指数衰减率               |
| adam_beta2   | 二阶矩估计的指数衰减率               |
| adam_epsilon | 指定优化器需要优化的参数              |
| decay_steps  | 衰减的步长                     |
| warmup_rate  | warmup 率                  |
| max_lr       | Adam 的初始最大学习率             |
| min_lr       | Adam 的初始最小学习率             |
| grad_clip    | 梯度裁剪范围，使用的是GlobalNorm梯度裁剪 |


### 并行维度

当前GPT模型已适配自动并行的**半自动策略**，用户可以通过配置文件选择并行的维度。

```yaml
  Distributed:
    dp_degree: 2
    mp_degree: 2
    pp_degree: 2
    sharding:
      sharding_degree: 1
      sharding_stage: 1
```

其中参数说明：

| **参数名**          | **参数释义**                             |
|------------------|--------------------------------------|
| dp_degree        | 数据并行维度                               |
| mp_degree        | 张量模型并行维度                             |
| pp_degree        | 流水线并行维度                              |
| sharding_degree  | 分组切分并行维度                             |
| sharding_stage   | 切分策略；1表示仅切分优化器状态，2表示再切分梯度，3表示再切分前向参数 |


### Engine训练控制

Engine训练设置完成模型训练/验证/推理等过程中的参数设置，是fleetX的AutoEngine的必要参数，所有使用该Engine都必须指定该配置。 其中包含的参数有：

```yaml
  Engine:
    max_steps: 500000
    num_train_epochs: 1
    accumulate_steps: 
    logging_freq: 1
    eval_freq: 500
    eval_iters: 10
    mix_precision:
      use_pure_fp16: True
      scale_loss: 32768.0
      custom_black_list: ["reduce_sum", "c_softmax_with_cross_entropy", "elementwise_div"]
      custom_white_list: ["lookup_table", "lookup_table_v2"]
    use_recompute: True
    save_load:
      save_steps: 1000
      output_dir: ./output
      ckpt_dir:
```

其中参数说明：
| **参数名**                      | **参数释义**               |
|------------------------------|------------------------|
| max_steps         | 最大训练步数                               |
| num_train_epochs  | 训练的epoch数量                           |
| accumulate_steps  | 梯度累加次数                           |
| logging_freq      | 训练日志打印的频率                            |
| eval_freq         | 模型评估间隔                               |
| eval_iters        | 模型评估时训练评估测试集的轮数                      |
| use_pure_fp16     | 是否使用purefp16精度训练                     |
| scale_loss        | 使用fp16精度下，loss的放缩比例                  |
| custom_black_list | 自定义算子黑名单。这个名单中的算子在支持float16计算时会被认为是数值危险的，它们的影响也可能会在下游操作中观察到。这些算子通常不会转为float16计算。 |
| custom_white_list | 自定义算子白名单。这个名单中的算子在支持float16计算时会被认为是数值安全的，并且对性能至关重要。如果设置了白名单，该名单中的算子会使用float16计算。|
| use_recompute     | 是否使用recompute训练                      |
| save_steps        | 保存模型间隔                               |
| output_dir        | 指定输出文件                               |
| ckpt_dir          | checkpoint的加载目录                      |


### 性能优化
```
待补充
```


## 运行方式
本目录给出32G V100环境下345M规模GPT模型半并行训练的策略配置如下：

| 模型规模   | 训练策略                    | yaml文件                      |
|----------|--------------------------------|------------------------------------------|
| 345MB    | fp16+dp8+recompute             | configs_345M_dp8.yaml                    |
| 345MB    | fp16+dp16+sharding16+recompute | configs_345M_dp16sharding16.yaml         |

若要在显存容量更小的16G V100环境下进行GPT大模型训练，可减小对应yaml文件中的`Model`-`hidden size`值。

### 策略支持

自动并行包括2种模式：半自动并行与全自动并行。
半自动并行包括了数据并行、张量模型并行、流水线并行和分组切片并行。此外还支持重计算、混合精度等策略，来减少显存占用、加速训练。**目前，GPT 模型训练可以支持任意维度的策略组合。**

|                 | data parallel | tensor parallel | pipeline parallel | pure fp16 | recompute |
|-----------------|---------------|-----------------|-------------------|-----------|-----------|
| sharding stage1 | ✓             | ✓               | ✓                 | ✓         | ✓         |
| sharding stage2 | ✓             | ✓               | ✓                 | ✓         | ✓         |
| sharding stage3 | ✓             | ✓               | ✓                 | ✓         | ✓         |


### 单机训练

以单机345M模型数据并行训练为例，通过``paddle.distributed.launch``启动多进程训练。

**启动命令**
```shell
log_dir=log_dp8
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" run_pretrain.py \
    -c ./configs_345M_dp8.yaml
```

若要在显存容量更小的16G V100环境下进行GPT模型单机训练，可通过减小`Model.hidden_size`调整模型规模至合适大小再启动训练，命令如下：


**启动命令**
```shell
log_dir=log_dp8
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" run_pretrain.py \
    -c ./configs_345M_dp8.yaml -o Model.hidden_size=1024
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

以2机16卡32G V100上的345M模型分组切分并行训练为例，启动命令为：

```shell
master_ip=master节点ip
master_port=可用的空闲端口号

log_dir=log_dp16sharding16
python -m paddle.distributed.launch --log_dir $log_dir --master=$master_ip:$master_port --nnodes=2 --devices "0,1,2,3,4,5,6,7" run_pretrain.py \
    -c ./configs_345M_dp16sharding16.yaml
```

若要在显存容量更小的16G V100环境下进行GPT模型两机训练，也可通过减小`Model.hidden_size`调整模型规模至合适大小再启动训练，命令如下：

```
master_ip=master节点ip
master_port=可用的空闲端口号

log_dir=log_dp16sharding16
python -m paddle.distributed.launch --log_dir $log_dir --master=$master_ip:$master_port --nnodes=2 --devices "0,1,2,3,4,5,6,7" run_pretrain.py \
    -c ./configs_345M_dp16sharding16.yaml \
    -o Model.hidden_size=1024
```

### 量化训练

```
待补充
```
