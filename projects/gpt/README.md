
## 推理部署
参考[文档](../../docs/inference.md)

## Benchmark
- 运行benchmark脚本
```
cd ppfleetx && python setup_cuda.py install && cd ..

python projects/gpt/benchmark.py --seq_len 128 --iter 10 --mp_size $MP_SIZE --model_dir ./output
```

| 模型          | 输入长度 | 输出长度 | batch size | GPU卡数 | FP16推理时延 | INT8推理时延 |
| :------------ | :------: | :------: | :--------: | :-----: | :----------: | :----------: |
| GPT-3(345M)   |    128   |    8     |     1      |    1    |   18.91ms    |   18.30ms    |
| GPT-3(345M)   |    128   |    8     |     2      |    1    |   20.01ms    |   18.88ms    |
| GPT-3(345M)   |    128   |    8     |     4      |    1    |   20.83ms    |   20.77ms    |
| GPT-3(345M)   |    128   |    8     |     8      |    1    |   24.06ms    |   23.90ms    |
| GPT-3(345M)   |    128   |    8     |    16      |    1    |   29.32ms    |   27.95ms    |
| GPT-3(6.7B)   |    128   |    8     |     1      |    1    |   84.93ms    |   63.96ms    |
| GPT-3(6.7B)   |    128   |    8     |     2      |    1    |   92.90ms    |   67.25ms    |
| GPT-3(6.7B)   |    128   |    8     |     4      |    1    |   107.27ms   |   78.98ms    |
| GPT-3(6.7B)   |    128   |    8     |     8      |    1    |   141.27ms   |   99.54ms    |
| GPT-3(6.7B)   |    128   |    8     |    16      |    1    |   209.64ms   |   140.97ms   |
| GPT-3(175B)   |    128   |    8     |     1      |    1    |   327.26ms   |   230.11ms   |
| GPT-3(175B)   |    128   |    8     |     2      |    1    |   358.61ms   |   244.23ms   |
| GPT-3(175B)   |    128   |    8     |     4      |    1    |   428.93ms   |   278.63ms   |
| GPT-3(175B)   |    128   |    8     |     8      |    1    |   572.49ms   |   344.00ms   |
| GPT-3(175B)   |    128   |    8     |    16      |    1    |   811.83ms   |   475.19ms   |

以上性能数据基于PaddlePaddle[每日版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-develop)
