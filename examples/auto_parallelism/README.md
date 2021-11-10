# GPT

## 模型介绍
GPT-[2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)/[3](https://arxiv.org/pdf/2005.14165.pdf) 是以[Transformer](https://arxiv.org/abs/1706.03762) 解码器为网络基本组件，使用自回归的方式在大规模无标注文本语料上进行预训练得到的语言生成模型。

本项目是语言模型 GPT 的 PaddlePaddle 实现。

## 快速开始

### 环境依赖
- regex
- sentencepiece
- tqdm
- visualdl

安装命令 `python3 -m pip install regex sentencepiece tqdm visualdl`

### 数据准备
为了方便用户运行测试本模型，本项目提供了处理好的300M的训练样本：

```shell
wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt/train.data.json_ids.npz
```

将所有预处理得到的npz文件统一放入一个文件夹中，以备训练使用：

```
mkdir data
mv train.data.json_ids.npz data
```

## 飞桨自动并行训练
飞桨自动并行训练，支持模型并行、流水线并行和数据并行策略，使得训练千亿参数规模的模型成为可能。在本示例中，我们提供了基于飞桨最新自动并行的GPT预训练模型。运行下面脚本，即可进行模型预训练对应模型：
```shell
sh run.sh "0,1" "./output/dp" --global_batch_size 4 --mp_degree 1 --dp_degree 2 --pp_degree 1
```
参数解释：
"0,1": 指定对应的gpu卡
"./output/dp": 指定日志输出路径
--micro_batch_size: 指定dp或者pp的batch size
--global_batch_size: 指定总的batch size
--mp_degree: 指定mp对应的卡数
--dp_degree: 指定dp对应的卡数
--pp_degree: 指定pp对应的卡数
需要注意的是，本示例支持的mp_degree、dp_degree和pp_degree的值为1或者2。
