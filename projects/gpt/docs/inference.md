
# 推理部署

模型训练完成后，可使用飞桨高性能推理引擎Paddle Inference通过如下方式进行推理部署。

## 1. 模型导出

以`GPT-3(345M)`模型为例，通过如下方式下载PaddleFleetX发布的训练好的权重。若你已下载或使用训练过程中的权重，可跳过此步。

```bash
mkdir -p ckpt
wget -O ckpt/GPT_345M.tar.gz https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
tar -xzf ckpt/GPT_345M.tar.gz -C ckpt/
```

通过如下方式进行推理模型导出
`GPT-3(345M)` 模型导出与推理
导出单卡`GPT-3(345M)`模型：
```bash
sh projects/gpt/auto_export_gpt_345M_single_card.sh
```

`GPT-3(6.7B)` 模型导出与推理
导出单卡`GPT-3(6.7B)`模型：
```bash
sh projects/gpt/auto_export_gpt_6.7B_mp1.sh
```

`GPT-3(175B)` 模型导出与推理
导出8卡`GPT-3(175B)`模型：
```bash
sh projects/gpt/auto_export_gpt_175B_mp8.sh
```


## 2. 推理部署

模型导出后，可通过`tasks/gpt/inference.py`脚本进行推理部署。
`GPT-3(345M)` 推理
```bash
bash projects/gpt/inference_gpt_345M_single_card.sh
```
`GPT-3(6.7B)` 推理
```bash
bash projects/gpt/inference_gpt_6.7B_single_card.sh
```
## 3. Benchmark
- 运行benchmark脚本
```
cd ppfleetx/ops && python setup_cuda.py install && cd ../..
bash projects/gpt/run_benchmark.sh
```

| 模型          | 输入长度 | 输出长度 | batch size | GPU卡数 | FP16推理时延 | INT8推理时延 |
| :------------ | :------: | :------: | :--------: | :-----: | :----------: | :----------: |
| GPT-3(345M)   |    128   |    8     |     1      |    1    |   18.91ms    |   18.30ms    |
| GPT-3(345M)   |    128   |    8     |     2      |    1    |   20.01ms    |   18.88ms    |
| GPT-3(345M)   |    128   |    8     |     4      |    1    |   20.83ms    |   20.77ms    |
| GPT-3(345M)   |    128   |    8     |     8      |    1    |   24.06ms    |   23.90ms    |
| GPT-3(345M)   |    128   |    8     |    16      |    1    |   29.32ms    |   27.95ms    |
| GPT-3(6.7B)   |    128   |    8     |     1      |    1    |   84.93ms    |   63.96ms    |
| GPT-3(6.7B)   |    128   |    8     |     2      |    1    |   91.93ms    |   67.25ms    |
| GPT-3(6.7B)   |    128   |    8     |     4      |    1    |   105.50ms   |   78.98ms    |
| GPT-3(6.7B)   |    128   |    8     |     8      |    1    |   138.56ms   |   99.54ms    |
| GPT-3(6.7B)   |    128   |    8     |    16      |    1    |   204.33ms   |   140.97ms   |
| GPT-3(175B)   |    128   |    8     |     1      |    8    |   327.26ms   |   230.11ms   |
| GPT-3(175B)   |    128   |    8     |     2      |    8    |   358.61ms   |   244.23ms   |
| GPT-3(175B)   |    128   |    8     |     4      |    8    |   428.93ms   |   278.63ms   |
| GPT-3(175B)   |    128   |    8     |     8      |    8    |   554.28ms   |   344.00ms   |
| GPT-3(175B)   |    128   |    8     |    16      |    8    |   785.92ms   |   475.19ms   |

以上性能数据基于PaddlePaddle[每日版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-develop) ，依赖CUDA 11.6测试环境。
