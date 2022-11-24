# MoCo
![MoCo](https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png)

This is a PaddlePaddle implementation of the 
[MoCov1](https://arxiv.org/abs/1911.05722), 
[MoCov2](https://arxiv.org/abs/2003.04297).


## Install Preparation

MoCo requires `PaddlePaddle >= 2.4`.
```shell
# git clone https://github.com/PaddlePaddle/PaddleFleetX.git
cd /path/to/PaddleFleetX
```

All commands are executed in the `PaddleFleetX` root directory.

```shell
python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

## Data Preparation

The imagenet 1k dataset needs to be prepared first and will be organized into the following directory structure.

```shell
ILSVRC2012
├── train/
├── xxx
├── val/
└── xxx
```

Then configure the path.

```shell
mkdir -p dataset
ln -s /path/to/ILSVRC2012 dataset/ILSVRC2012
```

## Unsupervised Training

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, you can run the script: 

### MoCo V1 (Single Node with 8 GPUs)
```shell
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    tools/train.py -c ppfleetx/configs/vis/moco/mocov1_pt_in1k_1n8c.yaml
```

### MoCo V2 (Single Node with 8 GPUs)
```shell
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    tools/train.py -c ppfleetx/configs/vis/moco/mocov2_pt_in1k_1n8c.yaml
```


The differences between MoCo v1 and MoCo v2 are as follows:
* MoCo v2 has a projector
* Data augmentation
* Softmax temperature
* Learning rate scheduler

## Linear Classification

When the unsupervised pre-training is complete, or directly download the provided pre-training checkpoint, you can use the following script to train a supervised linear classifier.

### MoCo v1

#### [Optional] Download checkpoint
```shell
mkdir -p pretrained/moco/
wget -O ./pretrained/moco/mocov1_pt_imagenet2012_resnet50.pdparams https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov1_pt_imagenet2012_resnet50.pdparams
```

#### Linear Classification Training (Single Node with 8 GPUs)

```shell
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    tools/train.py -c ppfleetx/configs/vis/moco/moco_lincls_in1k_1n8c.yaml \
    -o Model.model.base_encoder.pretrained=./pretrained/moco/mocov1_pt_imagenet2012_resnet50

```

### MoCo v2

#### [Optional] Download checkpoint
```shell
mkdir -p pretrained/moco/
wget -O ./pretrained/moco/mocov2_pt_imagenet2012_resnet50.pdparams https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov2_pt_imagenet2012_resnet50.pdparams
```

#### Linear Classification Training (Single Node with 8 GPUs)

```shell
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    tools/train.py -c ppfleetx/configs/vis/moco/moco_lincls_in1k_1n8c.yaml \
    -o Model.model.base_encoder.pretrained=./pretrained/moco/mocov2_pt_imagenet2012_resnet50

```

## Models

| Model   | Phase                 | Epochs | Top1 Acc | Checkpoint                                                   | Log                                                          |
| ------- | --------------------- | ------ | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MoCo v1 | Unsupervised Training | 200    | -        | [download](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov1_pt_imagenet2012_resnet50.pdparams) | [log](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov1_pt_imagenet2012_resnet50.log) |
| MoCo v1 | Linear Classification | 100    | 0.606141 | [download](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov1_lincls_imagenet2012_resnet50.pdparams) | [log](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov1_lincls_imagenet2012_resnet50.log) |
| MoCo v2 | Unsupervised Training | 200    | -        | [download](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov2_pt_imagenet2012_resnet50.pdparams) | [log](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov2_pt_imagenet2012_resnet50.log) |
| MoCo v2 | Linear Classification | 100    | 0.676595 | [download](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov2_lincls_imagenet2012_resnet50.pdparams) | [log](https://paddlefleetx.bj.bcebos.com/model/vision/moco/mocov2_lincls_imagenet2012_resnet50.log) |


## Citations

```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}

@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```
