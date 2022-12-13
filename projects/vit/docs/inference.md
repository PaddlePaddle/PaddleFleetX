# 推理部署

模型训练完成后，可使用飞桨高性能推理引擎Paddle Inference通过如下方式进行推理部署。

```bash
sh projects/vit/run_inference_base_patch16_224.sh
```
分解步骤如下：

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

模型导出后，可通过`projects/vit/inference.py`脚本进行推理部署。

```bash
python projects/vit/inference.py -c ppfleetx/configs/vis/vit/ViT_base_patch16_224_inference.yaml
```