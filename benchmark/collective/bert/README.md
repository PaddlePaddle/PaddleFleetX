# Bert Benchmarks


BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(https://arxiv.org/abs/1810.04805) is tested on 
selected multi-gpu environment for reference to users in PaddlePaddle community. 

Training with NVIDIA-DGX-1 (NVIDIA® Volta® V100)

## Benchmark Under Various Configurations

For some reasons, BERT can be trained with various hyperparameters during training. For example if we generate a batch through total number of tokens given a fixed maximum sequence length, the actual batch size would be different given various length of sentence pairs. If we generate a batch through a fixed batch size, total number of tokens would be differernt given various length of sentence pairs. We list benchmark under different training hyperparameters for long-term comparison. In the table, **t2048m128** means there are at most 2048 tokens in a batch and the max sequence length of a sentence pair is 128. Metric value in the table is steps/s. **2.05 * 2** means 2.05 steps/s per gpu card and the training is with two gpu card.

| GPUs 	| t2048m128 	| t2048m512 	| t4096m128 	| t4096m512 	| t8192m128 	| t8192m512 	|
|:----:	|:---------:	|:---------:	|:---------:	|:---------:	|:---------:	|:---------:	|
|   1  	|           	|           	|           	|           	|           	|    OOM    	|
|   2  	|  2.06 * 2 	|  1.79 * 2 	|  1.16 * 2 	|  0.91 * 2 	|  0.61 * 2 	|    OOM    	|
|   4  	|  2.06 * 4 	|  1.76 * 4 	|  1.16 * 4 	|  0.9 * 4  	|           	|    OOM    	|
|   8  	|  2.06 * 8 	|  1.70 * 8 	|  1.16 * 8 	|  0.87 * 8 	|           	|    OOM    	|
|  16  	|           	|           	|           	|           	|           	|    OOM    	|
|  32  	|           	|           	|           	|           	|           	|    OOM    	|
|  64  	|           	|           	|           	|           	|           	|    OOM    	|

We also compare efficiency given fixed batch size which is used in original paper. In the table below, **b16m128** means we 

## Scaling Efficiency Under Various Configurations



# Environment
* Instance type: NVIDIA® DGX-1™
* GPU: 8x NVIDIA® Volta® V100
* OS: Linux 3.10.0_3-0-0-17 #181 SMP Thu Feb 8 16:34:08 CST 2018 x86_64 GNU/Linux
* CUDA / cuDNN / NCCL : 9.2 / 7.3 / 2.3.4
* PaddlePaddle GitHub hash: b1e174e
* Fleet GitHub hash: 9165a70
* Build Command: cmake -DCMAKE_INSTALL_PREFIX=./output/ -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=ON -DWITH_MKL=OFF -DWITH_GPU=ON -DWITH_FLUID_ONLY=ON -DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python2.7/ -DPYTHON_LIBRARY=$PYTHONROOT/lib/libpython2.7.so -DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python2.7 ..
* Disk: Local SSD
* DataSet: English Wikipedia
* Test Date: June 2019
