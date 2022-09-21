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

## 简介

PaddleFleetX旨在打造一套简单易用、性能领先、且功能强大的端到端大模型工具库，覆盖大模型环境部署、数据处理、预训练、微调、模型压缩、推理部署全流程，并支持语言、视觉、多模态等多个领域的前沿大模型算法。


## 最新消息 🔥

**更新 (2022-09-21):** PaddleFleetX 发布 v0.1 版本.


## 安装

我们推荐从[预编译docker镜像](docs/quick_start.md#11-docker-环境部署)开始使用 PaddleFleetX，其中已经安装好了所有环境依赖。

如果您倾向根据自己的喜好安装环境，请根据以下的安装指导进行安装。

### 环境说明

* PaddleFleetX 依赖 GPU 版本的 [PaddlePaddle](https://www.paddlepaddle.org.cn/) ，请在使用前确保 PaddlePaddle 已经正确安装。
* 其他的 PyPI 依赖参见 `requirements.txt`。

### 安装 PyPI 依赖

请使用以下命令获取 PaddleFleetX 代码和安装依赖：

```shell
git clone https://github.com/PaddlePaddle/PaddleFleetX.git

cd PaddleFleetX
python -m pip  install -r requirements.txt
```

通过 [模型训练](./docs/quick_start.md#2-模型训练) 快速体验 PaddleFleetX 以及熟悉使用。

## 教程

* [快速开始](./docs/quick_start.md)
* 训练
  * [GPT](projects/gpt/docs/README.md)
  * [VIT](projects/vit/README.md)
  * [Imagen](projects/imagen/)
  * [Ernie](projects/ernie/)
* [推理](./docs/inference.md)
* [开发规范](./docs/standard.md)
* [集群部署](./docs/cluster_deployment.md)
* [部署常见问题](./docs/deployment_faq.md)


## 模型库

| **模型** | **参数量** | **预训练文件** |
|---------|-----------|---------------|
| GPT | 345M |  [GPT_345M](http://fleet.bj.bcebos.com/pretrained/gpt/GPT_345M_300B_DP_20220826.tgz)  |

## 性能



## 工业级应用



## 许可
PaddleFleetX 基于 [Apache 2.0 license](./LICENSE) 许可发布。


## 引用

```
@misc{paddlefleetx,
    title={PaddleFleetX: An Easy-to-use and High-Performance One-stop Tool for Deep Learning},
    author={PaddleFleetX Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleFleetX}},
    year={2022}
}
```
