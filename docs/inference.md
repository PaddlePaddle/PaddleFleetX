# PaddleFleetX 预测部署

**PaddleFleetX**提供了**Paddle Inference**高性能服务器端部署方式，支持Python端一键式模型导出和推理部署能力。

---

## 目录

- [1. 环境准备](#1)
- [2. 模型导出](#2)
- [3. 推理部署](#3)
- [4. 相关文档](#4)


<a name="1"></a>
## 1. 环境准备

### 1.1 安装PaddlePaddle

目前**PaddleFleetX**依赖**PaddlePaddle** 版本 `>=2.4`，请参考[PaddlePaddle安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)安装PaddlePaddle 2.4版本或[每日版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-develop)

建议使用飞桨官方Docker镜像运行PaddleFleetX，可参考[快速开始](./quick_start.md)文档配置和使用Docker

### 1.2 安装PaddleFleetX

通过以下命令下载PaddleFleetX最新代码

```bash
git clone https://github.com/PaddlePaddle/PaddleFleetX.git
cd PaddleFleetX
```

通过以下命令安装PaddleFleetX依赖库

```bash
python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

<a name="2"></a>
## 2. 模型导出

使用Paddle Inference高性能推理引擎，首先需要将模型导出为用于部署的推理模型，可通过`tools/export.py`进行模型导出，通过`-c`指定需要导出的模型的配置文件，通过`-o Engine.save_load.ckpt_dir=`指定导出模型时使用的权重。

以`GPT-3(345M)`模型为例，通过如下方式下载PaddleFleetX发布的训练好的权重。若你已下载或使用训练过程中的权重，可跳过此步。

```bash
mkdir -p ckpt
wget -O ckpt/GPT_345M.tar.gz https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
tar -xzf ckpt/GPT_345M.tar.gz -C ckpt/
```

通过如下方式进行推理模型导出

```bash
python tools/export.py \
    -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
    -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/
```

导出的模型默认保存在`./output`目录，可通过配置文件中`Engine.save_load.output_dir`或通过`-o Engine.save_load.output_dir=`指定


<a name="3"></a>
## 3. 推理部署

模型导出后，可通过`tasks/gpt/inference.py`脚本进行推理部署。

```bash
python tasks/gpt/inference.py -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml
```

<a name="4"></a>
## 4. 相关文档

### 4.1 模型支持

目前PaddleFleetX中支持高性能推理部署及依赖PaddlePaddle版本如下如下:

| 模型      | 依赖Paddle版本 |  支持部署精度  |
| :-------- | :------------: | :------------: |
| GPT-3     |     >=2.4      | FP32/FP16/INT8 |
| ERNIE     |    develop     |      FP32      |
| ViT       |     >=2.4      | FP32/FP16/INT8 |

### 4.2 Benchmark文档

模型推理Benchmark见[Benchmark文档](./inference_benchmark.md)
