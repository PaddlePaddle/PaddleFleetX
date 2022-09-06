# GPT

## 模型介绍
GPT-[2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)/[3](https://arxiv.org/pdf/2005.14165.pdf) 是以[Transformer](https://arxiv.org/abs/1706.03762) 解码器为网络基本组件，使用自回归的方式在大规模无标注文本语料上进行预训练得到的语言生成模型。

本项目是语言模型 GPT 的 PaddlePaddle 大模型实现。下是本例的简要目录结构及说明：

```text
.
├── tools.py                               # 训练参数配置处理脚本
├── gpt_module.py                          # GPT模型执行Module定义
├── gpt_config.py                          # GPT模型配置文件定义
├── single                                 # 单卡模型训练
    ├── configs_345m_single_card.yaml      # 单卡345M模型配置文件
    ├── configs_1.3B_single_card.yaml      # 单卡1.3B模型配置文件
    ├── run_pretrain.py                    # 单卡预训练入口
    ├── run.sh                             # 单卡训练启动入口
├── hybrid_parallel                        # 分布式模型训练
    ├── configs_1.3B_dp8.yaml              # 单机1.3B数据并行模型配置文件
    ├── configs_6.7B_sharding16.yaml       # 两机6.7B分组切片并行模型配置文件
    ├── configs_175B_mp8_pp16.yaml         # 16机175B混合并行模型配置文件
    ├── run_pretrain.py                    # 分布式预训练入口
    ├── run.sh                             # 分布式训练启动入口
├── auto_parallel                          # 自动并行分布式模型训练
    ├── configs_345M_dp8.yaml              # 单机345M数据并行模型配置文件
    ├── run_pretrain.py                    # 自动并行分布式预训练入口
    ├── run.sh                             # 自动并行分布式训练启动入口
```

## 快速开始

### 环境依赖

请确保已根据根目录 requirements.txt 安装所需依赖，或者通过以下命令快速安装

```shell
python -m pip install -r https://raw.githubusercontent.com/PaddlePaddle/FleetX/develop/requirements.txt -i https://mirror.baidu.com/pypi/simple
```

### 数据准备

数据获取和制作详见[GPT 模型预训练数据准备流程](https://github.com/PaddlePaddle/FleetX/tree/develop/fleetx/data/data_tools/gpt)

为了方便用户运行测试本模型，此处提供处理好的300M的训练样本，在单卡训练或混合并行训练前都需要通过以下命令获取数据。

**数据下载命令**
```shell
cd single # 或者 cd hybrid_parallel, 或者 cd auto_parallel

# 下载样例数据
mkdir data && cd data
wget -O gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget -O gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz

cd .. # 回到 single/hybrid_parallel/auto_parallel 路径下
```

### 模型训练

除了单卡训练，飞桨还支持数据并行、混合并行、自动并行、重计算等多种分布式策略，减少显存占用、加速训练，达到大模型可训练且训得快的效果。在模型训练前，需要根据模型规模选择合适的并行策略。下面分别从单卡训练、混合并行训练和自动并行训练三个方面来介绍GPT模型训练的配置文件和启动方式。


- [单卡训练](./single/README.md)

- [混合并行训练](./hybrid_parallel/README.md)

- [自动并行训练](./auto_parallel/README.md)

### 文本生成

- [单卡预训练模型文本生成](./single/README.md)

- [混合并行预训练模型文本生成](./hybrid_parallel/README.md)


## 参考文献
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
- [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)
