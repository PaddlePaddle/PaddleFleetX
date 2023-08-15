<p align="center">
  <img src="./paddlefleetx-logo.png" align="middle"  width="500" />
</p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleFleetX/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleFleetX?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleFleetX/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleFleetX?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PaddleFleetX/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleFleetX?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleFleetX/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleFleetX?color=ccf"></a>
</p>

## 简介

飞桨大模型套件PaddleFleetX是基于PaddlePaddle深度学习框架的大模型开发套件，旨在提供高性能、灵活易用大模型工具套件，用户可轻易灵活定制百亿和千亿大模型训练, PaddleFleetX大模型套件在『训』『压』『推』三大方向具备以下特性：
- 统一界面的高性能大模型分布式预训练和精调；4D并行分布式是PaddleFleetX核心基础框架能力，通过4种策略组合最大发挥训练硬件的计算和通讯能力，同时使用大模型Trainer统一大模型训练范式，提升大模型分布式框架易用性；特色高效精调PEFT结合4D并行策略，打破大模型训练资源限制，提升大模型训练吞吐
- 自研Shift-SmoothQuant无损压缩量化算法开源；结合自适应平衡激活和自动超参搜索等策略可以进行无损压缩量化，大幅提升大模型预测吞吐，打破预测硬件显存限制
- 高性能并行融合大模型推理服务引擎发布；支持量化模型推理，结合算子融合等特性支持大模型高性能推理，结合大模型不定长输出特性支持动态插入、流式输出等特性，有效提升大模型部署预测吞吐

![image](https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/8f569df3-bb85-4384-8ee8-30f808dcefc5)


## PaddleFleetX特色能力介绍
### 分布式训练框架：参数化配置复杂4D混合并行
百度飞桨在2021年发布了业界首创的4D混合并行策略，分别是张量并行，流水线并行，分片并行，数据并行，通过4D策略组合可以大幅提升大模型训练吞吐，有效降低大模型过程中的通信瓶颈，飞桨业内首创4D混合并行策略在国际权威的AI benchmark  MLPerf Training v2.0和v2.1 **世界第一**。


<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/1d8658f3-449f-4fac-927d-5a5210c39ce5" alt="4D并行" width="350" height="250">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/b6ff6ff1-42cb-433f-87b0-b8bd1d06d9aa" alt="Trainer配置" width="300" height="250">
</p>

同时4D并行策略相对复杂，不同的策略组合对整体训练流程有较大影响，相对数据并行传统的分布式策略而言接入成本高，因此在PaddleFleetX套件中的统一使用PaddleNLP Trainer来灵活配置分布式策略，同时结合Transformer API 和 4D并行策略设计实现，可以在Transformers API上无感使用4D并行策略，快速使用4D分布式策略。

### 大模型训练策略：内置高效精调算法，可单机精调千亿模型
目前大模型训练硬件瓶颈导致大模型训练难度高，同时PEFT(Parameter-Efficient Fine-Tuning) 可以打破大模型训练的硬件瓶颈，PEFT的原理是在训练过程中对大部分参数不做训练，而是对新增模型输入或者参数的进行训练，来有效降低大模型训练的资源占用，同时PEFT在大部分情况下不会降低SFT的训练的效果。同时在PaddleFleetX通过集成4D并行策略和LoRA、P-tuning等精调策略，可以在单机情况下(A100 80G  * 8)精调千亿模型，并且能提升训练吞吐；同时为了降低PEFT策略在Transformers API的使用难度，通过LoRA Config 和 Prefix Config 可以轻易将LLM模型转化LoRA Model 和 Prefix Model，并且可以和4D 并行策略完全耦合，快速上手PEFT训练策略。
               

<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/3d25e8e7-77fe-49d7-b4df-ed51e61f56b0" alt="PEFT配置" width="350" height="250">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/07135425-5e30-4fec-80da-c005d9133df7" alt="性能对比" width="300" height="250">
</p>

                
### 大模型压缩算法：自研Shift-SmoothQuant算法全面实现主流大模型无损量化
由于大模型预测成本高，对算力、显存要求高，同时在NLU模型相关上通过模型压缩量化可以做到无损量化，因此在NLG的生成模型做压缩量化有益于大模型的预测部署；PaddleFleetX 通过自研的Shift-SmoothQuant算法有效提升量化的精度和稳定性，通过 Shift 算法可以参数分布缩放到对称分布，同时通过 SmoothQuant 将异常参数值进行缩放合理范围内，因此通过 Shift-SmoothQuant 算法可以提升压缩的算法精度和稳定性；在 C-Eval 和 NL2SQL 两个benchmark数据集上在主流开源模型可以做到无损量化。

<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/2214d4eb-efe9-45b4-b540-32d9b9e10985" alt="PEFT配置" width="500" height="300">
   <img src="https://github.com/wawltor/PaddleFleetX/assets/16698950/ccbfafe4-0a5d-472b-ad88-e844a1b44468" alt="PEFT配置" width="500" height="300">

</p>



              
### 大模型推理引擎：基于算子融合的高性能大模型推理引擎
大模型推理部署在很多场景下会耗费较多的预测部署的资源，因此在PaddleFleetX在预测推理引擎中做了大量核心算子的fusion的过程，例如MultiHeadAttention、FFN，同时对CacheKV做了显存的预分配工作，减少生成过程中不断显存分配；同时由于LLM生成模型在生成过程中会同个batch会有生成长度不一致的问题，因此飞桨推理引擎支持动态Batch插入，适时替换更新不同的Query到预测的batch中，来提升预测吞吐。通过大量的预测优化，PaddleNLP动态图推理吞吐在主流模型相对HuggingFace提升200%+，静态图推理吞吐在主流推理产品中排名第一，下图是具体的对比细节。

<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/10872ada-a629-473c-bf17-e01192165e4d" alt="PEFT配置" width="500" height="300">
</p>



## PaddleFleetX使用
### 大语言模型使用链接 🔗[链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm)

### 跨模态大模型使用链接 🔗[链接](https://github.com/PaddlePaddle/PaddleMIX)

## 许可
PaddleFleetX 基于 [Apache 2.0 license](./LICENSE) 许可发布。

## 引用

```
@misc{paddlefleetx,
    title={PaddleFleetX: An Easy-to-use and High-Performance One-stop Tool for Deep Learning},
    author={PaddleFleetX Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleFleetX}},
    year={2022}
}
```
