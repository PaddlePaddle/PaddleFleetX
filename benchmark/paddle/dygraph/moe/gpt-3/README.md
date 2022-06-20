# benchmark of MoE for gpt-3 in dygraph mode
### 代码准备
```
# 下载安装 paddle2.3

# 拉取最新版本PaddleNLP库
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/examples/language_model/moe/dygraph/
```

### 运行命令示例
##### 1. DP with MoE
```bash 
bash benchmark_DPMoE.sh 
```

##### 2. Sharding with MoE
```bash
bash benchmark_ShardingMoE.sh
```

