# GPT 3D并行模型训练

当训练超大模型时，就必须借助3D混合并行策略，3D策略分别指数据并行，张量模型并行和流水线并行。其中张量模型并行将网络中的张量（Tensor）切分到不同的设备，从而降低单个设备的显存消耗；流水线并行将模型的不同层放置到不同的计算设备，降低单个计算设备的显存消耗；数据并行保存完整的模型参数并独立处理一份子数据集，以加速模型训练过程。联合三种训练方式，实现更大模型的训练。具体策略以及相关FleetAPI介绍可以参考以下教程：

- [张量模型并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/model_parallel_cn.html
)
- [流水线并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/pipeline_parallel_cn.html)

- [数据并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/data_parallel/index_cn.html)


## 3D混合并行
当前GPT模型已适配3D混合并行，并能够在训练超大模型，用户可以通过配置文件选择并行的维度。

```yaml
  Distributed:
    dp_degree: 1
    mp_degree: 8
    pp_degree: 16
```

其中参数含义：
- `dp_degree` 数据并行维度
- `mp_degree` 张量模型并行维度
- `pp_degree` 流水线并行维度


## 运行方式


以单机8卡为例，通过``paddle.distributed.launch``启动多进程训练。

```shell
log_dir=dp2_pp2_mp2
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" run_pretrain.py \
    -c ./configs.yaml
```

## 运行日志

模型切分信息：
```
2022-07-27 12:38:08,674-INFO: [pp_layers.py:192:__init__] Start Recompute for 2022-07-27 12:39:52,176-INFO: [pp_layers.py:320:_segment_network] start segment network..
2022-07-27 12:39:52,177-INFO: [pp_layers.py:327:_segment_network] segment result:0, 3, 7
2022-07-27 12:39:52,177-INFO: [pp_layers.py:337:_segment_network] stage=0, global_rank=0 ,layer_number=3
2022-07-27 12:39:52,177-INFO: [pp_layers.py:340:_segment_network] 0: EmbeddingPipe(vocab_size=51200, hidden_size=1024, hidden_dropout_prob=0.1, max_position_embeddings=1024, type_vocab_size=16, initializer_range=0.02)
2022-07-27 12:39:52,177-INFO: [pp_layers.py:340:_segment_network] 1: TransformerDecoderLayer(d_model=1024, nhead=2, dim_feedforward=4096, dropout=0.1, activation=gelu, attn_dropout=0.1, act_dropout=0.1, weight_attr=<paddle.fluid.param_attr.ParamAttr object at 0x7fde827de110>, bias_attr=None, num_partitions=2)
2022-07-27 12:39:52,177-INFO: [pp_layers.py:340:_segment_network] 2: TransformerDecoderLayer(d_model=1024, nhead=2, dim_feedforward=4096, dropout=0.1, activation=gelu, attn_dropout=0.1, act_dropout=0.1, weight_attr=<paddle.fluid.param_attr.ParamAttr object at 0x7fde827e2290>, bias_attr=None, num_partitions=2)
2022-07-27 12:39:52,177-INFO: [pp_layers.py:337:_segment_network] stage=1, global_rank=0 ,layer_number=4
2022-07-27 12:39:52,177-INFO: [pp_layers.py:340:_segment_network] 3: TransformerDecoderLayer(d_model=1024, nhead=2, dim_feedforward=4096, dropout=0.1, activation=gelu, attn_dropout=0.1, act_dropout=0.1, weight_attr=<paddle.fluid.param_attr.ParamAttr object at 0x7fde827e29d0>, bias_attr=None, num_partitions=2)
2022-07-27 12:39:52,177-INFO: [pp_layers.py:340:_segment_network] 4: TransformerDecoderLayer(d_model=1024, nhead=2, dim_feedforward=4096, dropout=0.1, activation=gelu, attn_dropout=0.1, act_dropout=0.1, weight_attr=<paddle.fluid.param_attr.ParamAttr object at 0x7fde827e2b90>, bias_attr=None, num_partitions=2)
2022-07-27 12:39:52,177-INFO: [pp_layers.py:340:_segment_network] 5: LayerNorm(normalized_shape=1024)
2022-07-27 12:39:52,177-INFO: [pp_layers.py:340:_segment_network] 6: EmbeddingPipe(vocab_size=51200, hidden_size=1024, hidden_dropout_prob=0.1, max_position_embeddings=1024, type_vocab_size=16, initializer_range=0.02)
2022-07-27 12:39:52,177-INFO: [pp_layers.py:346:_segment_network] loss: GPTPretrainingCriterionPipe
```

执行日志：

```
[2022-07-27 12:40:00,833] [    INFO] - global step 1, epoch: 0, batch: 0, loss: 11.083364487, avg_reader_cost: 0.23392 sec, avg_batch_cost: 2.05344 sec, speed: 0.49 step/s, ips_total: 3989 tokens/s, ips: 499 tokens/s, learning rate: 5.55556e-09
[2022-07-27 12:40:00,967] [    INFO] - global step 2, epoch: 0, batch: 1, loss: 11.054802895, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.13314 sec, speed: 7.51 step/s, ips_total: 61529 tokens/s, ips: 7691 tokens/s, learning rate: 8.33333e-09
[2022-07-27 12:40:01,075] [    INFO] - global step 3, epoch: 0, batch: 2, loss: 11.082803726, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.10738 sec, speed: 9.31 step/s, ips_total: 76289 tokens/s, ips: 9536 tokens/s, learning rate: 1.11111e-08
[2022-07-27 12:40:01,185] [    INFO] - global step 4, epoch: 0, batch: 3, loss: 11.070495605, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.10999 sec, speed: 9.09 step/s, ips_total: 74480 tokens/s, ips: 9310 tokens/s, learning rate: 1.38889e-08
[2022-07-27 12:40:01,298] [    INFO] - global step 5, epoch: 0, batch: 4, loss: 11.086440086, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.11292 sec, speed: 8.86 step/s, ips_total: 72548 tokens/s, ips: 9068 tokens/s, learning rate: 1.66667e-08
[2022-07-27 12:40:01,402] [    INFO] - global step 6, epoch: 0, batch: 5, loss: 11.046300888, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.10367 sec, speed: 9.65 step/s, ips_total: 79022 tokens/s, ips: 9878 tokens/s, learning rate: 1.94444e-08
[2022-07-27 12:40:01,506] [    INFO] - global step 7, epoch: 0, batch: 6, loss: 11.099355698, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.10319 sec, speed: 9.69 step/s, ips_total: 79385 tokens/s, ips: 9923 tokens/s, learning rate: 2.22222e-08
[2022-07-27 12:40:01,621] [    INFO] - global step 8, epoch: 0, batch: 7, loss: 11.076607704, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.11502 sec, speed: 8.69 step/s, ips_total: 71223 tokens/s, ips: 8903 tokens/s, learning rate: 2.50000e-08
[2022-07-27 12:40:01,726] [    INFO] - global step 9, epoch: 0, batch: 8, loss: 11.076778412, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.10425 sec, speed: 9.59 step/s, ips_total: 78577 tokens/s, ips: 9822 tokens/s, learning rate: 2.77778e-08
```
