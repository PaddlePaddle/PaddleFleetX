# PaddlePaddle Fast ImageNet Training

PaddlePaddle Fast ImageNet can train ImageNet dataset with fewer epochs. We implemented the it according to the blog 
[Now anyone can train Imagenet in 18 minutes](https://www.fast.ai/2018/08/10/fastai-diu-imagenet/) which published on the [fast.ai] website.
PaddlePaddle Fast ImageNet using the dynmiac batch size, dynamic image size, rectangular images validation and etc... so that the Fast ImageNet can achieve the baseline accuracy
(acc1: 75.9%, acc5: 93.0%) faster than the standard ResNet50.

## Experiment

1. Prepare the training data, resize the images to 160 and 352 using `resize.py`, the prepared data folder should look like:
    ``` text
    `-ImageNet
      |-train
      |-validation
      |-160
        |-train
        `-validation
      `-352
        |-train
        `-validation
    ```
1. Launch the training job: `python -m paddle.distributed.launch --selected_gpus="0,1" train.py --data_dir /data/imagenet`

