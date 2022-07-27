# GPT

## 模型介绍
GPT-[2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)/[3](https://arxiv.org/pdf/2005.14165.pdf) 是以[Transformer](https://arxiv.org/abs/1706.03762) 解码器为网络基本组件，使用自回归的方式在大规模无标注文本语料上进行预训练得到的语言生成模型。

本项目是语言模型 GPT 的 PaddlePaddle 大模型实现。下是本例的简要目录结构及说明：

```text
.
├── tools.py                # 训练参数配置处理脚本
├── single                  # 单卡模型（345M）
    ├── configs.yaml        # 模型配置文件
    ├── run_pretrain.py     # 预训练入口
    ├── run.sh              # 训练启动入口
├── data_parallel           # 数据并行模型文件（1.3B）
    ├── configs.yaml        # 模型配置文件
    ├── run_pretrain.py     # 预训练入口
    ├── run.sh              # 训练启动入口
├── group_shard             # group shard模型（6.7B）
    ├── configs.yaml        # 模型配置文件
    ├── run_pretrain.py     # 预训练入口
    ├── run.sh              # 训练启动入口
├── 3D_parallelism          # 3D混合并行模型（175B）
    ├── configs.yaml        # 模型配置文件
    ├── run_pretrain.py     # 预训练入口
    ├── run.sh              # 训练启动入口
```

## 快速开始

### 环境依赖

- regex
- sentencepiece >= 0.1.94
- tqdm
- visualdl
- paddlepaddle-gpu >= 2.2rc
- pybind11
- lac (可选)
- zstandard (可选)

安装命令 `pip install regex sentencepiece tqdm visualdl pybind11 lac zstandard`。
注：需要PaddlePaddle版本大于等于2.2rc，或者使用最新develop版本，安装方法请参见Paddle[官网](https://www.paddlepaddle.org.cn)。

### 数据准备

#### 数据获取与制作


数据获取和制作详见[GPT 模型预训练数据准备流程](https://github.com/PaddlePaddle/FleetX/tree/develop/fleetx/data/data_tools/gpt)

### 模型训练


根据不同的模型大小，需要选择不同的并行策略。常见的分布式策略有几种：[数据并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/data_parallel/index_cn.html)，[分组切分并行（group shard）](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/group_sharded_parallel_cn.html)，3D混合并行。



- [单卡训练](./single/README.md)

- 数据并行

- [分组切分并行]

- [3D混合并行](./3D_parallelism/README.md)


## 参考文献
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
- [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)
