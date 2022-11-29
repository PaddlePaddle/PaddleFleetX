# Vision Transformer

This project implements the (Vision Transformer) proposed by google [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).


## How to pretrain from scratch on imagenet2012

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

## Finetune ViT-B/16

### [Optional] Download checkpoint
```shell
mkdir -p pretrained/vit/
wget -O ./pretrained/vit/imagenet2012-ViT-B_16-224.pdparams https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-224.pdparams
```


### Finetune on imagenet2012
Finetune is similar to pre-training on ImageNet2012 dataset, we have provided the configured yaml file.

```shell
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c ppfleetx/configs/vis/vit/ViT_base_patch16_384_ft_in1k_2n16c_dp_fp16o2.yaml
```

### Finetune on cifar10

Note: CIFAR10 dataset is automatically downloaded and cached.

```shell
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c ppfleetx/configs/vis/vit/ViT_base_patch16_384_ft_cifar10_1n8c_dp_fp16o2.yaml
```

### Quantization Aware Training on ImageNet2012


```shell
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py \
    -c ppfleetx/configs/vis/vit/ViT_base_patch16_384_ft_qat_in1k_2n16c_dp_fp16o2.yaml \
    -o Model.model.drop_rate=0.0 \
    -o Data.Train.sampler.batch_size=16 \
    -o Optimizer.lr.learning_rate=5e-05 \
    -o Optimizer.weight_decay=0.0002 
```
量化训练的参数详细介绍见[模型压缩介绍](../../../docs/compression.md)。


## Model

| Model    | Phase    | Size   | Dataset      | Resolution | GPUs        | Img/sec | Top1 Acc | Pre-trained checkpoint                                                                             | Fine-tuned checkpoint | Log                                                                                      |
|----------|----------|--------|--------------|------------|-------------|---------|----------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| ViT-B_16 | pretrain | 167MiB | ImageNet2012 | 224        | A100*N2C16  | 7350    | 74.75%   | [download](https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-224.pdparams) | -                                                                                               | [log](https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-224.log) |
| ViT-B_16 | finetune | 167MiB | ImageNet2012 | 384        | A100*N2C16  | 1580    | 77.68%   | [download](https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-224.pdparams) | [download](https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-384.pdparams)          | [log](https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-384.log) |
| ViT-L_16 | finetune | 582MiB | ImageNet2012 | 384        | A100*N2C16  | 519     | 85.13%   | [download](https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet21k-jax-ViT-L_16-224.pdparams) | [download](https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet21k+imagenet2012-ViT-L_16-384.pdparams)          | [log](https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet21k+imagenet2012-ViT-L_16-384.log) |
| Quantized ViT-B_16 | finetune | 167MiB | ImageNet2012 | 384         | A100*N2C16  | 1580     |  77.71%  | [download](https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-384.pdparams) | [download](https://paddlefleetx.bj.bcebos.com/model/vision/vit/quantized_imagenet2012-ViT-B_16-384.pdparams)          | [log](https://paddlefleetx.bj.bcebos.com/model/vision/vit/quantized_imagenet2012-ViT-B_16-384.log) |



# 推理部署

模型训练完成后，可使用飞桨高性能推理引擎Paddle Inference通过如下方式进行推理部署。

## 1. 模型导出

首先将模型导出为用于部署的推理模型，可通过`tools/export.py`进行模型导出，通过`-c`指定需要导出的模型的配置文件，通过`-o Engine.save_load.ckpt_dir=`指定导出模型时使用的权重。

以`VIT-224`模型为例，通过如下方式下载PaddleFleetX发布的训练好的权重。若你已下载或使用训练过程中的权重，可跳过此步。

```bash
mkdir -p ckpt
wget -O ckpt/model.pdparams https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-224.pdparams
```

通过如下方式进行推理模型导出

```bash
python tools/export.py \
    -c ppfleetx/configs/vis/vit/ViT_base_patch16_224_inference.yaml \
    -o Engine.save_load.ckpt_dir=./ckpt/
```

导出的模型默认保存在`./output`目录，可通过配置文件中`Engine.save_load.output_dir`或通过`-o Engine.save_load.output_dir=`指定


## 2. 推理部署

模型导出后，可通过`projects/vit/inference_vit.py`脚本进行推理部署。

```bash
python projects/vit/inference_vit.py -c ppfleetx/configs/vis/vit/ViT_base_patch16_224_inference.yaml
```
