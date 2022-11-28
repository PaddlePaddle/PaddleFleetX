
# 推理部署

模型训练完成后，可使用飞桨高性能推理引擎Paddle Inference通过如下方式进行推理部署。

## 1. 模型导出

<!-- 首先将模型导出为用于部署的推理模型，可通过`tools/export.py`进行模型导出，通过`-c`指定需要导出的模型的配置文件，通过`-o Engine.save_load.ckpt_dir=`指定导出模型时使用的权重。 -->
首先将模型导出为用于部署的推理模型，可通过`projects/gpt/auto_export_gpt_***.sh`进行模型导出，通过`-c`指定需要导出的模型的配置文件。

以`GPT-3(345M)`模型为例，通过如下方式下载PaddleFleetX发布的训练好的权重。若你已下载或使用训练过程中的权重，可跳过此步。

```bash
mkdir -p ckpt
wget -O ckpt/GPT_345M.tar.gz https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
tar -xzf ckpt/GPT_345M.tar.gz -C ckpt/
```

通过如下方式进行推理模型导出

### `GPT-3(345M)` 模型导出与推理
导出单卡`GPT-3(345M)`模型：
```bash
sh projects/gpt/auto_export_gpt_345M_mp1.sh
```

### `GPT-3(6.7B)` 模型导出与推理
导出单卡`GPT-3(6.7B)`模型：
```bash
sh projects/gpt/auto_export_gpt_6.7B_mp1.sh
```

### `GPT-3(175BB)` 模型导出与推理
导出单卡`GPT-3(175B)`模型：
```bash
sh projects/gpt/auto_export_gpt_175B_mp8.sh
```


## 2. 推理部署

模型导出后，可通过`tasks/gpt/inference.py`脚本进行推理部署。

```bash
python projects/gpt/inference.py --mp_size $MP_SIZE --model_dir output
```