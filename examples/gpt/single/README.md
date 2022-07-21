# GPT-3 千亿参数模型训练

## 模型介绍
GPT-[3](https://arxiv.org/pdf/2005.14165.pdf) 是以[Transformer](https://arxiv.org/abs/1706.03762) 为基础的语言生成模型。GPT-3模型的最大参数量可以达到170B，如此大规模参数的模型对于训练使用的深度学习框架是一个巨大的挑战。


## 使用方法

### 环境依赖

- regex
- sentencepiece >= 0.1.94
- paddlepaddle-gpu >= 2.2rc

安装命令 `pip install regex sentencepiece tqdm visualdl`。
注：需要PaddlePaddle版本大于等于2.2rc，或者使用最新develop版本，安装方法请参见Paddle[官网](https://www.paddlepaddle.org.cn)。


### 数据获取与制作

[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/)是一个开源的英文网页文本数据集，数据来源于Reddit，经过去重、清洗、提取，最终包含800多万个文档。
本示例采用EleutherAI清洗好的[OpenWebText2数据](https://openwebtext2.readthedocs.io/en/latest/index.html#download-plug-and-play-version)

下载以后通过以下命令解压：

```shell
wget https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar
tar -xvf openwebtext2.json.zst.tar -C  /path/to/openwebtext
```

然后使用[data_tools](./data_tools)工具下的`create_pretraining_data.py`脚本进行数据集制作：
```
python -u  create_pretraining_data.py \
    --model_name gpt2-en \
    --tokenizer_name GPTTokenizer \
    --data_format JSON \
    --input_path /path/to/openwebtext/ \
    --append_eos \
    --output_prefix gpt_openwebtext  \
    --workers 40 \
    --log_interval 10000
```
处理时间约一个小时左右，就可以得到我们需要的`gpt_openwebtext_ids.npy`, `gpt_openwebtext_idx.npz`数据集文件。

为了方便用户运行测试本模型，本项目提供了处理好的300M的训练样本：
```shell
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```
mkdir data
mv gpt_en_dataset_300m_ids.npy ./data
mv gpt_en_dataset_300m_idx.npz ./data
```


```shell
cd static # 或者 cd dygraph
# 下载样例数据
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
cd ..
# 运行pretrian 脚本
sh run.sh
```
下面以静态图的运行脚本为例，说明训练参数的具体作用：
```shell
python run_pretrain.py \
    --input_dir "./data"\
    --output_dir "output"\
    --vocab_size 50304\
    --hidden_size 1024\
    --num_layers 16\
    --num_attention_heads 8\
    --max_seq_len 1024\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --device gpu\
    --eval_freq 1000\
    --warmup_rate 0.01\
    --scale_loss 32768\
    --global_batch_size 16\
    --micro_batch_size 8\
    --use_pure_fp16 True
```

其中参数释义如下：
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `output_dir` 指定输出文件。
- `vocab_size` 训练词表大小。
- `hidden_size` 隐藏层大小。
- `num_layers` transformer层数
- `num_attention_heads` attention head的数量。
- `max_seq_len` 输入文本序列的长度。
- `micro_batch_size` 单卡单次的 batch size大小。即单张卡运行一次前向网络的 batch size大小。
- `global_batch_size` 全局的batch size大小，即一次参数更新等效的batch size。
- `max_lr` 训练学习率。
- `min_lr` 学习率衰减的最小值。
- `max_steps` 最大训练步数。
- `save_steps` 保存模型间隔。
- `weight_decay` 权重衰减参数。
- `warmup_rate` 学习率warmup参数。
- `grad_clip` 梯度裁剪范围。
- `logging_freq` 日志输出间隔。
- `eval_freq` 模型评估间隔。
- `device` 训练设备。
- `use_pure_fp16` 使用pureFp16训练。
