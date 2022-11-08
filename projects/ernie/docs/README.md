# ERNIE: Enhanced Representation through kNowledge IntEgration


## 1. 模型简介

ERNIE是百度开创性提出的基于知识增强的持续学习语义理解框架，它将大数据预训练与多源丰富知识相结合，通过持续学习技术，不断吸收海量文本数据中词汇、结构、语义等方面的知识，实现模型效果不断进化。

ERNIE在情感分析、文本匹配、自然语言推理、词法分析、阅读理解、智能问答等16个公开数据集上全面显著超越世界领先技术，在国际权威的通用语言理解评估基准GLUE上，得分首次突破90分，获得全球第一。
相关创新成果也被国际顶级学术会议AAAI、IJCAI收录。
同时，ERNIE在工业界得到了大规模应用，如搜索引擎、新闻推荐、广告系统、语音交互、智能客服等。

ERNIE 通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于 BERT 学习原始语言信号，ERNIE 直接对先验语义知识单元进行建模，增强了模型语义表示能力。

这里我们举个例子：
```
Learnt by BERT ：哈 [mask] 滨是 [mask] 龙江的省会，[mask] 际冰 [mask] 文化名城。
Learnt by ERNIE：[mask] [mask] [mask] 是黑龙江的省会，国际 [mask] [mask] 文化名城。
```
在 BERT 模型中，我们通过『哈』与『滨』的局部共现，即可判断出『尔』字，模型没有学习与『哈尔滨』相关的任何知识。而 ERNIE 通过学习词与实体的表达，使模型能够建模出『哈尔滨』与『黑龙江』的关系，学到『哈尔滨』是 『黑龙江』的省会以及『哈尔滨』是个冰雪城市。


### 1.1 目录结构

```text
.
├── docs
│   └── README.md
├── pretrain_ernie_base_175B_mp8_pp16.sh    # 175B ernie-base模型，3D混合并行
├── pretrain_ernie_base_3D.sh               # ci测试
├── pretrain_ernie_base_6.7B_sharding16.sh  # 6.7B ernie-base模型，sharding16
├── pretrain_ernie_base.sh                  # 345M ernie-base模型，单卡
└── pretrain_ernie_large.sh                 # ernie-large模型，单卡     
```



### 1.2 依赖环境

- paddlenlp
- pybind11

安装命令 `pip install pybind11 paddlenlp`


## 2.中文预训练

ERNIE预训练采用的是MLM（Mask Language Model）的训练方式，采用WWM（Whole Word Mask）方式，对于完整语义单元的Token，会同时进行Mask。整体的训练损失loss是mlm_loss + sop_loss。


### 2.1 小规模语料预训练: 14GB - CLUECorpusSmall

<details>
<summary><b>CLUECorpusSmall 数据准备</b></summary>

#### 数据准备
数据下载部分请参考[data_tools](https://github.com/PaddlePaddle/PaddleFleetX/tree/develop/ppfleetx/data/data_tools/ernie/preprocess/docs/CLUECorpusSmall.md)目录，根据文档中`CLUECorpusSmall 数据集处理教程`，下载数据。下载好后:

解压文件
```shell
unzip comment2019zh_corpus.zip -d  clue_corpus_small_14g/comment2019zh_corpus
unzip news2016zh_corpus.zip    -d  clue_corpus_small_14g/news2016zh_corpus
unzip webText2019zh_corpus.zip -d  clue_corpus_small_14g/webText2019zh_corpus
unzip wiki2019zh_corpus.zip    -d  clue_corpus_small_14g/wiki2019zh_corpus
```
将txt文件转换为jsonl格式
```
python ./ppfleetx/data/data_tools/ernie/preprocess/trans_to_json.py  --input_path ./clue_corpus_small_14g --output_path clue_corpus_small_14g.jsonl
```
现在我们得到了jsonl格式的数据集，下面是针对训练任务的数据集应用，此处以ernie为例。
```
python -u  ./ppfleetx/data/data_tools/ernie/preprocess/create_pretraining_data.py \
    --model_name ernie-1.0-base-zh \
    --tokenizer_name ErnieTokenizer \
    --input_path clue_corpus_small_14g.jsonl \
    --split_sentences\
    --chinese \
    --cn_whole_word_segment \
    --cn_seg_func jieba \
    --output_prefix clue_corpus_small_14g_20220104 \
    --workers 48 \
    --log_interval 10000
```
数据共有文档`15702702`条左右，由于分词比较耗时，大概一小时左右可以完成。在当前目录下产出训练所需数据。
```
clue_corpus_small_14g_20220104_ids.npy
clue_corpus_small_14g_20220104_idx.npz
```

</details>


<details>
<summary><b>CLUECorpusSmall 开始训练</b></summary>

#### 开始训练


将制作好的数据`clue_corpus_small_14g_20220104_ids.npy,clue_corpus_small_14g_20220104_idx.npz`移动到input_dir中，即可开始训练。


除了单卡训练，飞桨还支持数据并行、混合并行、自动并行、重计算等多种分布式策略，减少显存占用、加速训练，达到大模型可训练且训得快的效果。在模型训练前，需要根据模型规模选择合适的并行策略。下面分别从单卡训练、混合并行训练和自动并行训练三个方面来介绍ERNIE模型训练的配置文件和启动方式。


- 单卡训练

```shell
cd PaddleFleetX # 如果已在 PaddleFleetX 根目录下，则忽略

# 345M
python tools/train.py -c ppfleetx/configs/nlp/ernie/pretrain_ernie_base_single_card.yaml 
```

- 混合并行

```shell
cd PaddleFleetX # 如果已在 PaddleFleetX 根目录下，则忽略

# 175B run_pretrain
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/ernie/pretrain_ernie_base_175B_mp8_pp16.yaml

```
