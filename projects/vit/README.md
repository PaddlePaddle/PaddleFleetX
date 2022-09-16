# Vision Transformer

This project implements the (Vision Transformer) proposed by google [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).


## How to pretrain from scratch on imagenet 1k

### Go to the main repo directory
All commands are executed in the home directory.
```
cd /path/to/PaddleFleetX
```

### Data
The imagenet 1k dataset needs to be prepared first and will be organized into the following directory structure.

```
ILSVRC2012
├── train/
├── train_list.txt
├── val/
└── val_list.txt
```

Then configure the path.

```shell
mkdir -p dataset
ln -s /path/to/ILSVRC2012 dataset/ILSVRC2012
```

### Train ViT-B/16

Note: ViT-B/16 needs run on 2 nodes with 16 A100 GPUs. If you only have a low-memory GPU, you can use gradient accumulation by setting `accumulate_steps` in yaml.


The following commands need to be run on each node.
```shell

python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c ppfleetx/configs/vis/vit/ViT_base_patch16_224_pt_in1k_2n16c_dp_fp16o2.yaml
```
