
# GPT模型蒸馏训练

本项目对语言模型 GPT 进行蒸馏。PaddleFleetX目前支持大模型蒸馏小模型，和模型自蒸馏两种模式，第一种比如GPT-1.3B蒸馏GPT-345M，模型自蒸馏比如GPT-345M蒸馏GPT-345M；下面是本例涉及的文件及说明：

```text
.
├── distill_gpt_345M_single_card.yaml            # GPT-1.3B教师网络蒸馏GPT-345M模型训练入口
├── distill_self_gpt_345M.sh                    # GPT-345M自蒸馏训练入口

```


### 环境依赖和数据准备
环境依赖和数据准备请参考[GPT训练文档](./README.md)。


### 预训练模型准备
蒸馏训练需加载[GPT-345M](https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz) 的预训练模型。

**预训练模型下载命令**

```shell
wget https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
tar xf GPT_345M.tar.gz
```

### GPT-345M模型自蒸馏

快速启动：
```shell
bash ./projects/gpt/distill_self_gpt_345M.sh
```

或如下启动：
```shell
export CUDA_VISIBLE_DEVICES=0,1

log_dir=log_kd
mkdir -p $log_dir

python -m paddle.distributed.launch \
    --log_dir $log_dir \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/distill_self_gpt_345M.yaml 
    
```

###  GPT-1.3B蒸馏GPT-345M模型

快速启动：

```shell
bash ./projects/gpt/distill_gpt_345M_single_card.yaml.sh
```

或如下启动：

```shell
export CUDA_VISIBLE_DEVICES=0,1

log_dir=log_kd
mkdir -p $log_dir

python -m paddle.distributed.launch \
    --log_dir $log_dir \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/distill_gpt_345M_single_card.yaml.yaml 
    
```

###  

