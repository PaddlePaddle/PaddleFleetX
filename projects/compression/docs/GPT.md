
# GPT模型压缩

本项目对语言模型 GPT 进行压缩。目前，PaddleFleetX 提供了 [GPT-345M量化模型](https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M_QAT_w_analysis.tar) 的预训练模型文件；基于 [LAMBADA](https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl)，采用 ACC(accuracy) 指标后的评估结果如下：

| **模型文件** | **数据类型** | **ACC** |
|---------|-----------|---------------|
| GPT-345M | FP16 |  44.17%  |
| GPT-345M | INT8 |  44.94%  |

下面是本例的简要目录结构及说明：

```text
.
├── qat_gpt_345M_single_card.sh            # 单卡345M模型量化训练入口
├── qat_gpt_345M_mp8.sh                    # 8卡345M模型模型并行量化训练入口
├── qat_gpt_6.7B_sharding16.sh             # 16卡6.7B模型分组切片并行量化训练入口
├── eval_qat_gpt_345M_single_card.sh       # 单卡345M量化模型验证入口
├── export_qat_gpt_345M_single_card.sh     # 单卡345M量化模型导出入口

```


### 环境依赖

请确保已根据根目录 requirements.txt 安装所需依赖，或者通过以下命令快速安装

```shell
python -m pip install -r https://raw.githubusercontent.com/PaddlePaddle/PaddleFleetX/develop/requirements.txt -i https://mirror.baidu.com/pypi/simple
```

### 数据准备

数据获取和制作详见[GPT 模型预训练数据准备流程](https://github.com/PaddlePaddle/PaddleFleetX/tree/develop/ppfleetx/data/data_tools/gpt)

为了方便用户运行测试本模型，此处提供处理好的300M的训练样本，在单卡训练或混合并行训练前都需要通过以下命令获取数据。

**数据下载命令**
```shell
cd PaddleFleetX # 如果已在 PaddleFleetX 根目录下，则忽略

# 下载样例数据
mkdir data && cd data
wget -O gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget -O gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz

cd .. # 回到 PaddleFleetX 根目录下
```

### 预训练模型准备
量化训练需加载[GPT-345M](https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz) 的预训练模型。

**预训练模型下载命令**
```shell
wget https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
tar xf GPT_345M.tar.gz
```

### 量化训练

- [345M模型单卡训练](./qat_gpt_345M_single_card.sh)

```shell
export CUDA_VISIBLE_DEVICES=0

log_dir=log_hybrid
rm -rf $log_dir

python ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/qat_gpt_345M_single_card.yaml \
    -o Engine.max_steps=100000 \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Optimizer.lr.decay_steps=72000 \
    -o Optimizer.weight_decay=0.02 \
    -o Optimizer.lr.max_lr=5.0e-6 \
    -o Optimizer.lr.min_lr=1.0e-6 \
    -o Compress.pretrained='./PaddleFleetX_GPT_345M_220826'
    
```

- [345M模型模型并行训练](./qat_gpt_345M_mp8.sh)

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

log_dir=log_hybrid
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/qat_gpt_345M_mp8.yaml \
    -o Engine.max_steps=100000 \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Optimizer.lr.decay_steps=72000 \
    -o Optimizer.weight_decay=0.02 \
    -o Optimizer.lr.max_lr=5.0e-6 \
    -o Optimizer.lr.min_lr=1.0e-6 \
    -o Compress.pretrained='./PaddleFleetX_GPT_345M_220826'
```

- [6.7B模型分组切片并行训练](./qat_gpt_6.7B_sharding16.sh)

```shell
log_dir=log_hybrid
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/qat_gpt_6.7B_sharding16.yaml \
    -o Engine.max_steps=100000 \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Optimizer.lr.decay_steps=72000 \
    -o Optimizer.weight_decay=0.02 \
    -o Optimizer.lr.max_lr=5.0e-6 \
    -o Optimizer.lr.min_lr=1.0e-6 \
    -o Compress.pretrained='./PaddleFleetX_GPT_345M_220826'

```
Tips：尽管设置的最大训练轮数为100000轮，但实验经验4000轮即可达到最优效果。


### 量化训练精度调优
针对生成式预训练语言模型的模型压缩一直是学界上的难点，潜在的原因目前并不清楚。经我们研究分析发现，生成式预训练语言模型的Transformer层的权重分布差异较大，且由于生成式预训练语言模型的从左到右预测的性质，量化误差会逐步累积，精度损失较大。为了保证量化模型的精度，PaddleSlim提供量化训练敏感度分析工具，可以有效定位模型某层带来的量化损失较大，以规避一些敏感层并提高量化模型精度。

PaddleSlim中的量化训练敏感度分析工具仅支持静态图模型，需要将量化模型导出为静态图模型。导出命令为：

```shell
# 下载未经过分析的量化模型
wget https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M_QAT_wo_analysis.tar
tar xf GPT_345M_QAT_wo_analysis.tar

export CUDA_VISIBLE_DEVICES=0

python ./tools/export.py \
    -c ./ppfleetx/configs/nlp/gpt/export_quantized_gpt_345M_single_card.yaml \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Engine.save_load.ckpt_dir='./GPT_345M_QAT_wo_analysis/'
```

具体步骤可参考
[GPT量化训练敏感度分析示例](https://github.com/PaddlePaddle/PaddleSlim/blob/example/quantization_analysis/GPT/README.md)。





### 模型验证
```shell
# 下载验证数据
wget https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl

# 下载已经训练好的量化模型
wget https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M_QAT_w_analysis.tar
tar xf GPT_345M_QAT_w_analysis.tar

export CUDA_VISIBLE_DEVICES=0

python ./tools/eval.py \
    -c ./ppfleetx/configs/nlp/gpt/eval_quantized_gpt_345M_single_card.yaml \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Engine.save_load.ckpt_dir='./GPT_345M_QAT_w_analysis'
    -o Offline_Eval.eval_path=./lambada_test.jsonl \
    -o Offline_Eval.cloze_eval=True 
```
