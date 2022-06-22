# Hybrid Parallel

[Introduction](#introduction)  
[Quick Start](#quick-start)  
[Model Examples](#model-examples)  

## Introduction
## Quick Start
### 1. Required Environment
- python3.6+
- paddle2.3+
- For using GPUs, you'll also need install CUDA and NCCL

### 2. PaddlePaddle Installation

select a version for your environment from the [downloading page](https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html)  
	 
we select `gpu-2.3.0.post110-cp37` as an example
```bash
wget https://paddle-wheel.bj.bcebos.com/2.3.0/linux/linux-gpu-cuda11.0-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.3.0.post110-cp37-cp37m-linux_x86_64.whl
python -m pip install paddlepaddle_gpu-2.3.0.post110-cp37-cp37m-linux_x86_64.whl
```

### 3. Model & Dataset & Dependencies Preparation
```
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP && git checkout baf322f20afbdd0fedc457c50da6a80e6ce87f2c
cd examples/language_model/gpt-3/dygraph

mkdir -p data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/train.data.json_ids.npz
cd ..

python -m pip install paddlenlp

```

### 4. Hybrid Parallel Environment Initialization

```python
from paddle.distributed import fleet
strategy = fleet.DistributedStrategy()

global_batch_size=16
accumulate_steps=2
DP=2
MP=2
PP=2
strategy.hybrid_configs = {
        "dp_degree": DP,
        "mp_degree": MP,
        "pp_degree": PP,
         }
if PP > 1:
    strategy.pipeline_configs = {
            "accumulate_steps": accumulate_steps,
            "micro_batch_size": global_batch_size // DP // accumulate_steps
    }

fleet.init(is_collective=True, strategy=strategy)
hcg = fleet.get_hybrid_communicate_group()

dp_group = hcg.get_data_parallel_group()
mp_group = hcg.get_model_parallel_group()
pp_group = hcg.get_pipe_parallel_group()

```

### 5. Splitng Weights if Needed
If you need to seperate linear's weights or embedding's weights into distributed gpus, you can replace `nn.Linear` with `fleet.meta_parallel.ColumnParallelLinear` or `fleet.meta_parallel.RowParallelLinear`, and replace `nn.Embedding` with `fleet.meta_parallel.VocabParallelEmbedding`.
Folowing are some samples:

- Linear weights
```python
from paddle import nn
#before spliting:
otuput = nn.Linear(
    row,
    col,
    weight_attr=weight_attr)

#after spliting along column:
output = fleet.meta_parallel.ColumnParallelLinear(
    row,
    col,
    weight_attr=weight_attr,
    has_bias=True,
    gather_output=False)

#after spliting along row:
output = fleet.meta_parallel.RowParallelLinear(
    row,
    col,
    weight_attr=weight_attr,
    has_bias=True,
    gather_output=False)
```
- Embedding weights
```python
#before spliting:
word_embeddings = nn.Embedding(
     (vocab_size,hidden_size),
     param_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
     mean=0.0, std=initializer_range)))

#after spliting:
word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
     vocab_size,
     hidden_size,
     weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
     mean=0.0, std=initializer_range)))

```

### 6. Distributed Model And Optimzier Preparation
```python
model = fleet.distributed_model(model)
optimizer = fleet.distributed_optimizer(optimizer)
```
### 7. Distributed Dataset Preparation
we use DistributedSampler as data sampler if data parallel is used
```python
from paddle.io import DistributedBatchSampler, DataLoader
batch_sampler = DistributedBatchSampler(
            dataset,
            batch_size=args.local_batch_size,
            num_replicas=DP,
            rank=hcg.get_data_parallel_rank())
data_loader = DataLoader(
            dataset=dataset,
            places=places,
            feed_list=data_holders,
            batch_sampler=batch_sampler)

```

### 8. Training Starting
Because the parallelism environment has been already initialized in this sample code, we can skip step 4,5,6,7 and directly start the train process using the following command:
```bash
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-small-en \
    --input_dir "./data"\
    --output_dir "output"\
    --global_batch_size 16\
    --micro_batch_size 2\
    --dp_degree 2\
    --mp_degree 2\
    --pp_degree 2
```

## Model Examples
|Task|Model|
|-----|-----|
|NLP  |[gpt-3](../../benchmark/paddle/dygraph/hybrid_parallelism/gpt-3/README.md)|
|NLP  |[MoE](../../benchmark/paddle/dygraph/moe/gpt-3/README.md)|
