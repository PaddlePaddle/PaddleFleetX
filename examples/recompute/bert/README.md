
# introduction

This is an example of BERT pretraining and fine-tuning with Recompute.

# Quick start: Fine-tuning on XNLI-en dataset

## download data
请分别下载 [XNLI dev/test set](https://bert-data.bj.bcebos.com/XNLI-1.0.zip) 和 [XNLI machine-translated training set](https://bert-data.bj.bcebos.com/XNLI-MT-1.0.zip)，然后解压到同一个目录。

## download pretrained model

请按此链接下载数据：[BERT-Large, Uncased](https://bert-models.bj.bcebos.com/uncased_L-24_H-1024_A-16.tar.gz)

## fine-tuning!

```shell
PRETRAINED_CKPT_PATH=uncased_L-24_H-1024_A-16/params
DATA_PATH=xnli
bert_config_path=uncased_L-24_H-1024_A-16/bert_config.json
vocab_path=uncased_L-24_H-1024_A-16/vocab.txt
sh train_cls.sh $PRETRAINED_CKPT_PATH $bert_config_path $vocab_path $DATA_PATH
```
## Results

Training context: V100 GPU Cards

- max batch size

When setting seq_len to 512, max batch size +328%

|Model|Baseline|Recompute|
|:---:|:---:|:---:|
|bert-large|28|120|
|bert-base|80|300|

When setting seq_len to 128, max batch size +510%

|Model|Baseline|Recompute|script|
|:---:|:---:|:---:|:---:|
|bert-large|93|562|scripts/bert_large_max_batch_size.sh|
|bert-base|273|1390|scripts/bert_base_max_batch_size.sh|

- Final test accuracy 

|Baseline|Recompute|
|:---:|:---:|
|85.24%|85.91%|

注：以上结果为4次实验的平均准确率, 由于训练由随机性，所以最终准确率有diff。

- Training speed -22.5%

|Baseline|Recompute|
|:---:|:---:|
|1.094 steps/s|0.848 steps/s|

- Estimated memory usage for Bert-large with batch size 72000:

![recompute](https://github.com/PaddlePaddle/Fleet/blob/develop/examples/recompute/bert/image/memory_anal.png)

