# GPT 单卡模型训练

## 运行方式

本文档按照345M和1.3B规模大小，给出32G V100环境下GPT模型单卡训练的策略配置如下：

| 模型规模 | 训练策略       | yaml文件                    | 显存占用 |
|----------|----------------|-------------------------------|----------|
| 345M     | fp16           | pretrain_gpt_345M_single_card.yaml | 30.9GB   |
| 1.3B     | fp16+recompute | pretrain_gpt_1.3B_single_card.yaml | 26.0GB   |

**启动命令**
```shell
cd FleetX # 如果已在 FleetX 根目录下，则忽略

# 345M
python tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml

# 1.3B
python tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_single_card.yaml
```

若要在显存容量更小的16G V100环境下进行GPT模型单机训练，可通过减小`Model.hidden_size`调整模型规模至合适大小，或使用重计算等显存优化策略再启动训练，命令如下：

```shell
# 345M
python tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Model.use_recompute=True

# 1.3B
python tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_single_card.yaml \
    -o Model.hidden_size=1024
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


# GPT 单卡模型评估

我们提供了对[WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)、[LAMBADA](https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl)两种数据集的评估脚本，其中数据集WikiText采用的是PPL(perplexity)评估指标，LAMBADA采用的是ACC(accuracy)指标。

## 参数释义

请在模型评估前将前述数据集下载到FleetX根目录下(WikiText数据集需要解压缩)，然后可以使用配置文件配置评估相关的参数，包括：

```yaml
  Offline_Eval:
    eval_path: ./wikitext-103/wiki.valid.tokens
    cloze_eval: False
    overlapping_eval: 32
    batch_size: 8
    max_seq_len: 1024
    logging_freq: 10
```

其中参数对应的释义如下：

| **参数名**                      | **参数释义**          |
|------------------------------|------------------------|
| eval_path         | 评估数据集地址                      |
| cloze_eval  | lambada数据集参数                     |
| overlapping_eval  | wikitext数据集参数              |
| batch_size         | 模型评估时batch size             |
| max_seq_len        | 模型评估时文本序列长度           |
| logging_freq     | 评估日志的打印频率                |

## 运行方式

以单卡345M模型评估为例，可以使用如下命令启动评估：

### WikiText数据集评估

```shell
cd FleetX # 如果已在 FleetX 根目录下，则忽略

mkdir -p ckpt
wget -O ckpt/GPT_345M_300B_DP_20220826.tgz http://fleet.bj.bcebos.com/pretrained/gpt/GPT_345M_300B_DP_20220826.tgz
tar -xzf ckpt/GPT_345M_300B_DP_20220826.tgz -C ckpt/

wget -O wikitext-103-v1.zip https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip -q wikitext-103-v1.zip

ckpt_dir=ckpt/GPT_345M_300B_DP_20220826/mp_00_sharding_00_pp_00/
eval_dir=./wikitext-103

python tools/eval.py -c ppfleetx/configs/nlp/gpt/eval_gpt_345M_single_card.yaml \
    -o Engine.save_load.ckpt_dir=$ckpt_dir \
    -o Offline_Eval.eval_path=$eval_dir/wiki.valid.tokens \
    -o Offline_Eval.overlapping_eval=32 \
    -o Offline_Eval.batch_size=16
```

评估日志如下：
```shell
[2022/09/15 12:22:42] ppfleetx INFO: [eval] epoch: 0, batch: 0, loss: 0.170368048, speed: 0.31 step/s
[2022/09/15 12:22:56] ppfleetx INFO: [eval] epoch: 0, batch: 10, loss: 0.231640193, speed: 0.75 step/s
[2022/09/15 12:23:09] ppfleetx INFO: [eval] epoch: 0, batch: 20, loss: 0.292417919, speed: 0.74 step/s
[2022/09/15 12:23:23] ppfleetx INFO: [eval] epoch: 0, batch: 30, loss: 0.351391476, speed: 0.74 step/s
[2022/09/15 12:23:36] ppfleetx INFO: [eval] epoch: 0, batch: 40, loss: 0.415404772, speed: 0.74 step/s
```

评估结果如下：

```shell
[2022/09/15 12:34:37] ppfleetx INFO: validation results on ./wikitext-103/wiki.valid.tokens | avg loss: 2.9554E+00 | ppl: 1.9210E+01 | adjusted ppl: 2.4948E+01 | token ratio: 1.0884484081583892
```

### LAMBADA数据集评估

```shell
cd FleetX # 如果已在 FleetX 根目录下，则忽略

mkdir -p ckpt
wget -O ckpt/GPT_345M_300B_DP_20220826.tgz http://fleet.bj.bcebos.com/pretrained/gpt/GPT_345M_300B_DP_20220826.tgz
tar -xzf ckpt/GPT_345M_300B_DP_20220826.tgz -C ckpt/

wget -O lambada_test.jsonl https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl

ckpt_dir=ckpt/GPT_345M_300B_DP_20220826/mp_00_sharding_00_pp_00/

python tools/eval.py -c ppfleetx/configs/nlp/gpt/eval_gpt_345M_single_card.yaml \
    -o Engine.save_load.ckpt_dir=$ckpt_dir \
    -o Offline_Eval.eval_path=./lambada_test.jsonl \
    -o Offline_Eval.cloze_eval=True \
    -o Offline_Eval.batch_size=16

```

评估日志如下：
```shell
[2022/09/15 12:29:25] ppfleetx INFO: [eval] epoch: 0, batch: 0, number correct: 50.000000000, speed: 0.30 step/s
[2022/09/15 12:29:38] ppfleetx INFO: [eval] epoch: 0, batch: 10, number correct: 130.000000000, speed: 0.76 step/s
[2022/09/15 12:29:52] ppfleetx INFO: [eval] epoch: 0, batch: 20, number correct: 209.000000000, speed: 0.76 step/s
[2022/09/15 12:30:05] ppfleetx INFO: [eval] epoch: 0, batch: 30, number correct: 279.000000000, speed: 0.76 step/s
[2022/09/15 12:30:18] ppfleetx INFO: [eval] epoch: 0, batch: 40, number correct: 343.000000000, speed: 0.75 step/s
```

评估结果如下：

```shell
[2022/09/15 12:46:29] ppfleetx INFO: validation results on ./lambada_test.jsonl | number correct: 2.1240E+03 | total examples: 5.1530E+03 | avg accuracy: 4.1219E-01
```

# GPT Zero-shot 文本生成

## 参数释义

```yaml
  Generation:
    top_k: 50
    top_p: 0.75
    temperature: 1.0
    min_dec_len: 1
    max_dec_len: 200
    num_return_sequences: 1
    decode_strategy: "sampling"
```

其中参数说明：

| **参数名**      | **参数释义**                  |
|--------------|---------------------------|
| top_k | 每次为采样挑选保留分数最高的 k 个 token        |
| top_p   | 如果设置小于 1.0 的小数，则保留加起来为 top_p 或更高的最可能的概率的 token。默认值为 1.0        |
| temperature   |  调节下一个 token 的概率温度，logits = logits / temperature，默认值为 1.0           |
| min_dec_len | 最小生成 token 长度              |
| max_dec_len  | 最大生成 token 长度                     |
| num_return_sequences  | 每个输入生成的序列个数，默认值为 1                  |
| decode_strategy       | 解码策略，默认值为 "sampling"，目前只支持 "sampling"，未来会支持 "greedy_search"，"beam_search" |

## 文本生成

下载预训练好的模型，快速体验文本生成

### 快速体验文本生成


```shell
cd FleetX # 如果已在 FleetX 根目录下，则忽略

mkdir -p ckpt
wget -O ckpt/GPT_345M_300B_DP_20220826.tgz http://fleet.bj.bcebos.com/pretrained/gpt/GPT_345M_300B_DP_20220826.tgz
tar -xzf ckpt/GPT_345M_300B_DP_20220826.tgz -C ckpt/

python tasks/gpt/generation.py \
    -c ppfleetx/configs/nlp/gpt/generation_gpt_345M_single_card.yaml \
    -o Engine.save_load.ckpt_dir=./ckpt/GPT_345M_300B_DP_20220826/mp_00_sharding_00_pp_00

# 生成的文本，由于 checkpoint 不同，超参不同，随机数不同，您执行可能会生成不一样的内容

Prompt: Hi, GPT2. Tell me who Jack Ma is.
Generation: Hi, GPT2. Tell me who Jack Ma is. I don’t want to hear that.”

For now, the only question the crowd is asking is whether or not Jack Ma will step down from the board of directors of Alibaba.

Jack Ma on why he never wanted to run for President in 2016:

There were two reasons. One is that I wanted to spend more time with my family. I thought it was better to spend more time with my family and spend more time with my children. So it was a very personal reason. But the second reason was that I thought it would be difficult to get elected, because there are a lot of political interests in this country. So I thought it was better to spend more time with my family.

On how Alibaba will evolve into a new player in China’s transportation and logistics sector:

I think that we are going to become a very important player in the logistics industry. So our strategy is to make it easy for people to travel.
```

### 剖析体验文本生成

#### GPT 文本生成模块初始化

```python
    module = build_module(cfg)
    module.model.eval()
```

#### 预训练模型加载

```python
    # 获取到预训练 checkpoint 的根目录
    ckpt_dir = cfg.Engine.save_load.ckpt_dir

    # 构造出具体路径
    model_path = os.path.join(ckpt_dir, "model.pdparams")

    # 加载模型参数
    model_dict = paddle.load(model_path)

    # FP16 模型参数转成 FP32 模型参数
    for key, value in model_dict.items():
        model_dict[key] = model_dict[key].astype(paddle.float32)

    # 设置模型参数为预训练参数
    module.model.set_state_dict(model_dict)
```

#### 文本生成与结果展示

```python
    input_text = "Historical Records: Tell us about the history of the Great Wall."
    result = module.generate(input_text)

    print(f'Prompt: {input_text}')
    print(f'Generation: {result[0]}')
```

### 模型导出与预测部署

#### 模型导出

如果需要进行模型预测部署，需要先导出用于线上部署的预测模型，可通过如下命令进行模型导出：

1. 下载预训练模型权重，如你已下载，可跳过此步

```shell
cd FleetX # 如果已在 FleetX 根目录下，则忽略

mkdir -p ckpt
wget -O ckpt/GPT_345M_300B_DP_20220826.tgz http://fleet.bj.bcebos.com/pretrained/gpt/GPT_345M_300B_DP_20220826.tgz
tar -xzf ckpt/GPT_345M_300B_DP_20220826.tgz -C ckpt/
```

2. 导出预测模型

```bash
python tools/export.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Engine.save_load.ckpt_dir=./ckpt/GPT_345M_300B_DP_20220826/mp_00_sharding_00_pp_00
```

导出的模型默认保存在`./output`目录，可通过配置文件中`Engine.save_load.output_dir`或通过`-o Engine.save_load.output_dir=`指定

导出脚本输出如下：

```bash
INFO:fleetx.core.engine.eager_engine:NOTE: disable use_pure_fp16 in export mode
INFO:fleetx.inference.export_utils:export inference model saved in ./output/rank_0/model
```

#### 预测部署

模型导出后，可以使用Paddle Inference高性能推理引擎完成模型的预测部署，可通过如下脚本和命令进行模型预测：

```bash
python tools/inference.py -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml
```

`tools/inference.py`模型从配置文件中`Inference.model_dir`中读取导出的预测模型，可通过`-o Inference.model_dir=`指定预测模型所在目录，默认为`./output`

预测脚本输出如下：

```bash
Prompt: Hi, GPT2. Tell me who Jack Ma is.
Generation: Hi, GPT2. Tell me who Jack Ma is.

GPT2: My name is Jack Ma.

Jack Ma: Jack Ma’s actually a man, he’s a Chinese businessman, born in 1927 in Fuzhou, China, he moved to Shanghai at a young age, he became an investment banker at the investment bank UBS, he then moved to San Francisco and became an entrepreneur.

He got involved in the stock market in the early ‘60s, he was very early on into this technology, he was the first investor in Yahoo and he had a very long career there, and he’s also the founder of Alibaba, the company that is responsible for the majority of China’s internet.

Peter McCormack: So it was a natural move for you to start getting involved in this area. You’re actually a fan of Bitcoin.

Jack Ma: Yes, I am.

Peter McCormack: You’ve invested in many Bitcoin
```
