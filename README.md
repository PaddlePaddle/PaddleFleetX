<p align="center">
  <img src="./paddlefleetx-logo.png" align="middle"  width="500" />
</p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleFleetX/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleFleetX?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleFleetX/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleFleetX?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PaddleFleetX/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleFleetX?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleFleetX/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleFleetX?color=ccf"></a>
</p>

## Features

[PaddleFleetX](https://github.com/PaddlePaddle/PaddleFleetX) is an open source repo for ...

TODO: A GIF showing the tasks supported by PaddleFleetX.

## Top News 🔥

**Update (2022-09-16):** PaddleFleetX v0.1 is released.


## Installation

We recommend to get started with PaddleFleetX using [pre-build container](docs/quick_start.md#11-docker-环境部署) which comes with all requirements installed.
If you prefer to install the requirements on your own, please follow this installation guide.

### Requirements

* [PaddlePaddle](https://www.paddlepaddle.org.cn/) GPU version must be installed **before** using PaddleFleetX.
* GPUs are required to work with PaddleFleetX, NVIDIA V100 or above are recommended. 
* Other PyPI requirements are listed in `requirements.txt`.

### Install

With PaddlePaddle well installed, you can fetch PaddleFleetX and install its dependencies with the following commands,

```shell
git clone https://github.com/PaddlePaddle/PaddleFleetX.git

cd PaddleFleetX
python -m pip  install -r requirements.txt
```

Check out the [quick start](./docs/quick_start.md#2-模型训练) for training examples and further usage.

## Tutorials

* [Quick Start](./docs/quick_start.md)
* [Modules](./docs/modules.md)
* [How to Training]()
  * [GPT](projects/gpt/docs/README.md)
  * [VIT](projects/vit/README.md)
  * [Imagen](projects/imagen/)
  * [Ernie](projects/ernie/)
* [How to Finetune]()
* [How to Inference](./docs/inference.md)
* [How to Develop by Yourself](./docs/standard.md)
* [Cluster Deployment](./docs/cluster_deployment.md)
* [Deployment FAQ](./docs/deployment_faq.md)


## Model Zoo
To download more useful pre-trained models see [model zoo]().

## Performance
TODO: Chart showing PaddleFleetX performance benefits.

For more performance see
* [GPT]()
* [ERNIE]()
* [ViT]()
* [Imagen]()

## Industrial Application
Coming soon.


## License

This project is released under the [Apache 2.0 license](./LICENSE).

## Citation

```
@misc{paddlefleetx,
    title={PaddleFleetX: An Easy-to-use and High-Performance One-stop Tool for Deep Learning},
    author={PaddleFleetX Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleFleetX}},
    year={2022}
}
```
