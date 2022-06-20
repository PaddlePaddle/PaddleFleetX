# benchmark of MoE for gpt-3 in dygraph mode
### 代码准备
```
# 下载安装 paddle2.3

# 拉取最新版本PaddleNLP库
git clone https://github.com/PaddlePaddle/PaddleNLP.git -b develop
cd PaddleNLP/examples/language_model/moe/dygraph/
git checkout 156b6761e88094f0fe6e10bd9698ee6d18f49759
```
### 性能数据
硬件环境： GPU A100
| Parameters(B) |  Attention heads  | Hidden size  | Vocab size |  Layers  | Experts  | GPUs  | Batch size | speed(tokens) | Memory(GB)  |
|---------------|-------------------|--------------|------------|----------|----------|-------|------------|---------------|-------------|
| 13.9          | 64                | 4096         | 50304      | 12       | 8        | 8     | 8          | 31085         | 56.8        |
| 26.8          | 64                | 4096         | 50304      | 12       | 16       | 16    | 16         | 59136         | 53.9        |
| 52.6          | 64                | 4096         | 50304      | 12       | 32       | 32    | 32         | 113456        | 54.5        |
| 104.1         | 64                | 4096         | 50304      | 12       | 64       | 64    | 64         | 209970        | 54.4        |
| 207.2         | 64                | 4096         | 50304      | 12       | 128      | 128   | 128        | 376968        | 54.3        |

### 运行命令示例
##### 1. DP with MoE
```bash 
bash benchmark_DPMoE.sh 
```

##### 2. Sharding with MoE
```bash
bash benchmark_ShardingMoE.sh
```

