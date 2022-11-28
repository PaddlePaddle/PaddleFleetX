# PaddleFleetX 推理Benchmark

---

## 目录

- [1. GPT-3](#1)
- [2. ERNIE](#2)
- [3. ViT](#3)


<a name="1"></a>
## 1. GPT-3

### 1.1 测试环境

- PaddleFleetX 2.4
- PaddlePaddle 2.4
- GPU: Nvidia Ampere A100(80G) * 8
- CPU: Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60GHz * 128core
- CUDA 11.2
- cuDNN 8.2

### 1.2 Benchmark

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

<a name="2"></a>
## 2. ERNIE

ERNIE模型Benchmark数据测试中

<a name="3"></a>
## 3. ViT

### 1.1 测试环境

- PaddleFleetX 2.4
- PaddlePaddle 2.4
- GPU: Nvidia Tesla T4(16G) * 1
- CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 20core
- CUDA 11.6
- cuDNN 8.4
- TensorRT 8.4

### 1.2 Benchmark

| 模型              | 输入尺寸 | batch size | GPU卡数 | FP16推理时延 | INT8推理时延 |
| :---------------- | :------: | :--------: | :-----: | :----------: | :----------: |
| ViT-base-patch16  |   224    |     1      |    1    |    2.79ms    |   2.41ms    |
| ViT-base-patch16  |   224    |     8      |    1    |   14.81ms    |   10.95ms    |
| ViT-base-patch16  |   224    |    16      |    1    |   29.55ms    |   21.63ms    |
| ViT-base-patch16  |   224    |    32      |    1    |   58.90ms    |   40.12ms    |
| ViT-base-patch16  |   384    |     1      |    1    |    8.91ms    |   -   |
| ViT-base-patch16  |   384    |     8      |    1    |   65.21ms    |   -   |
| ViT-base-patch16  |   384    |    16      |    1    |   131.32ms   |   -   |
| ViT-base-patch16  |   384    |    32      |    1    |   263.55ms   |   -   |
| ViT-large-patch16 |   224    |     1      |    1    |    8.30ms    |   -   |
| ViT-large-patch16 |   224    |     8      |    1    |   49.37ms    |   -   |
| ViT-large-patch16 |   224    |    16      |    1    |   95.10ms    |   -   |
| ViT-large-patch16 |   224    |    32      |    1    |   188.80ms   |   -   |
