# benchmark of hybrid_parallelism for gpt-3 in dygraph mode
### 代码准备
```
# 下载安装 paddle2.3

# 拉取最新版本PaddleNLP库
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/examples/language_model/gpt-3/dygraph

```
### 性能数据
硬件环境: GPU V100
| Parameters(B) |  Attention heads  | Hidden size  |  Layers  | MP | PP  | DP  | GPUs | Batch size | Achieved TFLOPS per GPU | Percentage of theoretical peak FLOPS  |
|---------------|-------------------|--------------|----------|----|-----|-----|------|------------|-------------------------|---------------------------------------|
| 9.66          | 16                | 4096         | 48       | 8  | 1   | 1   | 8    | 512        | 55.77                   | 44.61%                                |
| 16.31         | 16                | 6144         | 36       | 8  | 2   | 2   | 32   | 512        | 60.84                   | 48.67%                                |
| 91            | 32                | 8192         | 112      | 8  | 16  | 1   | 128  | 512        | 62.05                   | 49.64%                                |
| 146           | 32                | 11264        | 96       | 32 | 8   | 2   | 512  | 2048       | 60.82                   | 48.65%                                |


### 运行命令
参考[readme](https://github.com/sljlp/PaddleNLP/tree/develop/examples/language_model/gpt-3#readme)
