# GPT 单卡模型训练


## 参数释义

### 模型网络

网络部分完成了网络的组网操作，GPT在[FleetX/fleetx/models/gpt_model/modeling.py]([../../ppocr/modeling](https://github.com/PaddlePaddle/FleetX/tree/develop/fleetx/models/gpt_model))下。 
可以使用配置文件配置模型的规模，如：

```yaml
  Model:
    vocab_size: 50304
    hidden_size: 1024
    num_layers: 24
    num_attention_heads: 16
    ffn_hidden_size: 4096
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


GPT训练默认使用AdamW优化器以及cosine 学习率衰减，这里通过配置文件配置优化器的参数，如：

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

### 训练控制

通过配置文件配置训练相关的超参数，如：


```yaml
  device: gpu
  max_steps: 500000
  num_train_epochs: 1
  seed: 1024
  use_recompute: False
  recompute_granularity:
  batch_size:
    global_batch_size: 8
    local_batch_size: 8
    micro_batch_size: 8
  mix_precision:
    use_pure_fp16: True
    scale_loss: 32768.0
  logging_freq: 1
  eval_freq: 500
  eval_iters: 10
  dataset:
    input_dir: ./data
    split: '949,50,1'
    max_seq_len: 1024
  save_load:
    save_steps: 1000
    output_dir: ./output
    ckpt_dir: 
  fused_linear: False 
```

其中参数说明：

| **参数名**           | **参数释义**                             |
|-------------------|--------------------------------------|
| device            | 训练设备                                 |
| max_steps         | 最大训练步数                               |
| num_train_epochs  | 训练的epoch数量                           |
| seed              | 随机种子，保证训练过程可复现                       |
| use_recompute     | 是否使用recompute训练                      |
| recompute_granularity | recompute训练的粒度，可选 `full` `only_attn`，full即recompute全部transformer，only_attn表明只recompute self attention部分 |
| global_batch_size | 全局的batch size大小，即一次参数更新等效的batch size |
| local_batch_size  | 每个进程训练的batch size大小                  |
| micro_batch_size  | 每次前向计算的batch size大小                  |
| use_pure_fp16     | 是否使用purefp16精度训练                     |
| scale_loss        | 使用fp16精度下，loss的放缩比例                  |
| logging_freq      | 训练日志打印的频率                            |
| eval_freq         | 模型评估间隔                               |
| eval_iters        | 模型评估时训练评估测试集的轮数                      |
| input_dir         | 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件       |
| split             | 训练集，验证集和测试集的切分比例                     |
| max_seq_len       | 输入文本序列的长度                            |
| save_steps        | 保存模型间隔                               |
| output_dir        | 指定输出文件                               |
| ckpt_dir          | checkpoint的加载目录                      |
| fused_linear      | 是否使用fused_linear代替传统Linear加速训练。注：该功能需要cuda 11.6及以上编译的paddle支持。       |
| tensor_fusion | 是否使用tensor_fustion功能加速训练 |

## 运行方式

本目录中按照345M和1.3B规模大小，给出32G V100环境下GPT模型单卡训练的策略配置如下：

| 模型规模 | 训练策略       | yaml文件                    | 显存占用 |
|----------|----------------|-------------------------------|----------|
| 345M     | fp16           | configs_345m_single_card.yaml | 30.9GB   |
| 1.3B     | fp16+recompute | configs_1.3B_single_card.yaml | 26.0GB   |

**启动命令**
```shell
# 345M
python run_pretrain.py -c ./configs_345m_single_card.yaml

# 1.3B
python run_pretrain.py -c ./configs_1.3B_single_card.yaml
```

**运行日志**

```
[2022-07-27 12:42:46,601] [    INFO] - global step 1, epoch: 0, batch: 0, loss: 11.052017212, avg_reader_cost: 0.05710 sec, avg_batch_cost: 1.59627 sec, speed: 0.63 step/s, ips_total: 5132 tokens/s, ips: 5132 tokens/s, learning rate: 5.55556e-09
[2022-07-27 12:42:47,102] [    INFO] - global step 2, epoch: 0, batch: 1, loss: 11.030861855, avg_reader_cost: 0.00016 sec, avg_batch_cost: 0.50125 sec, speed: 2.00 step/s, ips_total: 16343 tokens/s, ips: 16343 tokens/s, learning rate: 8.33333e-09
[2022-07-27 12:42:47,600] [    INFO] - global step 3, epoch: 0, batch: 2, loss: 11.054017067, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.49697 sec, speed: 2.01 step/s, ips_total: 16484 tokens/s, ips: 16484 tokens/s, learning rate: 1.11111e-08
[2022-07-27 12:42:48,096] [    INFO] - global step 4, epoch: 0, batch: 3, loss: 11.027174950, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.49582 sec, speed: 2.02 step/s, ips_total: 16522 tokens/s, ips: 16522 tokens/s, learning rate: 1.38889e-08
[2022-07-27 12:42:48,591] [    INFO] - global step 5, epoch: 0, batch: 4, loss: 11.037425041, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.49529 sec, speed: 2.02 step/s, ips_total: 16540 tokens/s, ips: 16540 tokens/s, learning rate: 1.66667e-08
[2022-07-27 12:42:49,088] [    INFO] - global step 6, epoch: 0, batch: 5, loss: 11.038356781, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.49619 sec, speed: 2.02 step/s, ips_total: 16510 tokens/s, ips: 16510 tokens/s, learning rate: 1.94444e-08
[2022-07-27 12:42:49,582] [    INFO] - global step 7, epoch: 0, batch: 6, loss: 11.032723427, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.49402 sec, speed: 2.02 step/s, ips_total: 16582 tokens/s, ips: 16582 tokens/s, learning rate: 2.22222e-08
[2022-07-27 12:42:50,086] [    INFO] - global step 8, epoch: 0, batch: 7, loss: 11.025435448, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.50364 sec, speed: 1.99 step/s, ips_total: 16266 tokens/s, ips: 16266 tokens/s, learning rate: 2.50000e-08
[2022-07-27 12:42:50,583] [    INFO] - global step 9, epoch: 0, batch: 8, loss: 11.047873497, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.49669 sec, speed: 2.01 step/s, ips_total: 16493 tokens/s, ips: 16493 tokens/s, learning rate: 2.77778e-08
```
