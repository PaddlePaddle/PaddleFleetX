
# introduction

This is an example of BERT pretraining and fine-tuning with Recompute.

# Quick start: Fine-tuning on XNLI-en dataset

## download data
    Please follow this page to download xnli data: https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/PaddleLARK/BERT

Downloaded data
```shell
|-- multinli
|   |-- multinli.train.ar.tsv
|   |-- multinli.train.bg.tsv
|   |-- multinli.train.de.tsv
|   |-- multinli.train.el.tsv
|   |-- multinli.train.en.tsv
|   |-- multinli.train.es.tsv
|   |-- multinli.train.fr.tsv
|   |-- multinli.train.hi.tsv
|   |-- multinli.train.ru.tsv
|   |-- multinli.train.sw.tsv
|   |-- multinli.train.th.tsv
|   |-- multinli.train.tr.tsv
|   |-- multinli.train.ur.tsv
|   |-- multinli.train.vi.tsv
|   `-- multinli.train.zh.tsv
|-- README.md
|-- xnli
|   |-- xnli.dev.en.jsonl
|   |-- xnli.dev.en.tsv
|   |-- xnli.test.en.jsonl
|   `-- xnli.test.en.tsv
|-- xnli.dev.jsonl
|-- xnli.dev.tsv
|-- xnli.test.jsonl
`-- xnli.test.tsv
``` 

## download pretrained model

Please follow this page to download pretrained model named `BERT-Large, Uncased`

## fine-tuning!

```shell
PRETRAINED_CKPT_PATH=uncased_L-24_H-1024_A-16
DATA_PATH=xnli
bert_config_path=uncased_L-24_H-1024_A-16/bert_config.json
vocab_path=uncased_L-24_H-1024_A-16/vocab.txt
sh train_cls.sh $PRETRAINED_CKPT_PATH $bert_config_path $vocab_path $DATA_PATH
```
## Results

Baseline: 

- Max batch size

|Baseline|Recompute|
|:---:|:---:|
|12000|45000|

- Loss curve:

- Final test accuracy

- Training speed

|Baseline|Recompute|
|:---:|:---:|
|||

# Quich start: Pretraining
