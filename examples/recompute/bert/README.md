
# introduction

This is an example of BERT pretraining and fine-tuning with Recompute.

# Quick start: Fine-tuning on XNLI-en dataset

## download data
请分别下载 [XNLI dev/test set](https://bert-data.bj.bcebos.com/XNLI-1.0.zip) 和 [XNLI machine-translated training set](https://bert-data.bj.bcebos.com/XNLI-MT-1.0.zip)，然后解压到同一个目录。

## download pretrained model

请按此链接下载数据：[BERT-Large, Uncased](https://bert-models.bj.bcebos.com/uncased_L-24_H-1024_A-16.tar.gz)

## fine-tuning!

```shell
sh train_cls.sh 
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

# Quick start: Pretraining

```shell
sh pretrain.sh
```

## results

- Bert Large model

When setting seq_len to 512, the max batch size is increased by 300% compared with the Baseline, while the training speed is decresed by 31%.

|Model|Baseline|Recompute| Recompute + mixed precision| 
|:---:|:---:|:---:|:---:|
|batch size| 14 | 56 | 87 |
|speed|18.5 sents/s| 12.88 sents/s| 19.14 sents/s |

- Bert Base model

When setting seq_len to 512, the max batch size is increased by 245% compared with the Baseline, while the training speed is decresed by 32%.

|Model|Baseline|Recompute| Recompute + mixed precision| 
|:---:|:---:|:---:|:---:|
|batch size| 42 | 145 | 200 |
|speed|53.4 sents/s| 36.5 sents/s| 59.8 sents/s |

