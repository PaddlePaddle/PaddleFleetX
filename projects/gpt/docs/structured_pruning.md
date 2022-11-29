# GPT模型结构化稀疏

本项目对语言模型 GPT 进行结构化稀疏（以下简称稀疏）。

下面是本例涉及的文件及说明：

```text
.
├── prune_gpt_345M_single_card.sh             # 单卡345M稀疏训练入口
├── prune_gpt_6.7B_sharding16.sh              # 16卡6.7B模型分组切片并行稀疏训练入口
├── eval_pruned_gpt_345M_single_card.sh       # 单卡345M稀疏模型验证入口
├── export_pruned_gpt_345M_single_card.sh     # 单卡345M稀疏模型导出入口

```


### 环境依赖和数据准备
环境依赖和数据准备请参考[GPT训练文档](./README.md)。

特别的，本示例需要依赖 PaddleSlim develop版本。安装命令如下：

```shell
git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
pip install -r requirements.txt
python setup.py install
```


### 预训练模型准备
稀疏训练需加载[GPT-345M](https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz) 的预训练模型。

**预训练模型下载命令**
```shell
wget https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
tar xf GPT_345M.tar.gz
```

### 稀疏训练

- [345M模型稀疏训练](../gpt/prune_gpt_345M_single_card.sh)

快速启动：
```shell
bash ./projects/gpt/prune_gpt_345M_single_card.sh
```

或如下启动：
```shell
export CUDA_VISIBLE_DEVICES=0

log_dir=log_prune
rm -rf $log_dir

python ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/prune_gpt_345M_single_card.yaml \
    -o Engine.max_steps=100000 \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Optimizer.lr.decay_steps=7200 \
    -o Optimizer.weight_decay=0.0 \
    -o Optimizer.lr.max_lr=2.5e-5 \
    -o Optimizer.lr.min_lr=5.0e-6 \
    -o Compress.pretrained='./PaddleFleetX_GPT_345M_220826'
    
```

- [6.7B模型分组切片并行训练](../gpt/prune_gpt_6.7B_sharding16.sh)

快速启动：
```shell
bash ./projects/gpt/prune_gpt_6.7B_sharding16.sh
```

或如下启动：
```shell
log_dir=log_prune
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/prune_gpt_6.7B_sharding16.yaml \
    -o Engine.max_steps=500000 \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Optimizer.lr.decay_steps=72000 \
    -o Optimizer.weight_decay=0.0 \
    -o Optimizer.lr.max_lr=2.5e-5 \
    -o Optimizer.lr.min_lr=5.0e-6 \
    -o Compress.pretrained='./PaddleFleetX_GPT_6.7B'

```
Tips: 随着稀疏度的增加，训练轮数也建议增加。例如，在25%稀疏度下，15000 个step精度就会最优了；而50%稀疏度下，我们需要训练到 40000 到 500000 个steps 左右。

**注意**： 由于目前稀疏与 recompute 不兼容，会导致显存占用较大，所以上述脚本需要在80G显存的机器运行。

### 模型验证
```shell
# 下载验证数据
wget https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl

export CUDA_VISIBLE_DEVICES=0
python ./tools/eval.py \
    -c ./ppfleetx/configs/nlp/gpt/eval_pruned_gpt_345M_single_card.yaml \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Engine.save_load.ckpt_dir='./output'
    -o Offline_Eval.eval_path=./lambada_test.jsonl \
    -o Offline_Eval.cloze_eval=True
```

### 模型导出
```shell
export CUDA_VISIBLE_DEVICES=0
python ./tools/export.py \
    -c ./ppfleetx/configs/nlp/gpt/generation_pruned_gpt_345M_single_card.yaml \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Engine.save_load.ckpt_dir='./output'
```
