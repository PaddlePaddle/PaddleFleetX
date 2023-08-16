<p align="center">
  <img src="./paddlefleetx-logo.png" align="middle"  width="350" />
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

飞桨大模型套件PaddleFleetX是基于飞桨深度学习框架开发的大模型开发套件，旨在提供高性能、灵活易用大模型工具套件，用户可轻易灵活定制百亿和千亿大模型训练, 在**开发**、**训练**、**精调**、**压推**、**推理**、**部署**六大环节提供端到端全流程优化。

![image](https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/8f569df3-bb85-4384-8ee8-30f808dcefc5)

## 特色介绍

### 大模型训练：高效灵活的4D混合并行加速

飞桨在2021年发布了业界首创的4D混合并行策略，分别是张量并行，流水线并行，参数分组并行和数据并行，通过4D策略组合可以大幅提升大模型训练吞吐，有效降低大模型过程中的通信瓶颈。


<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/1d8658f3-449f-4fac-927d-5a5210c39ce5" alt="4D并行" width="40%" height="40%">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/b6ff6ff1-42cb-433f-87b0-b8bd1d06d9aa" alt="Trainer配置" width="29.5%" height="29.5%">
</p>

同时4D并行策略相对复杂，不同的策略组合对整体训练流程有较大影响，相对数据并行传统的分布式策略而言接入成本高，因此在PaddleFleetX套件中的统一的 Trainer 来灵活配置分布式策略，只需要简单的配置即可在大模型训练中快速生效。

### 大模型精调：内置高效精调算法，可单机精调千亿模型

目前大模型训练硬件瓶颈导致大模型训练难度高，同时PEFT(Parameter-Efficient Fine-Tuning) 可以打破大模型训练的硬件瓶颈，PEFT的原理是在训练过程中对大部分参数不做训练，而是对新增模型输入或者参数的进行训练，来有效降低大模型的训练门槛，PEFT在大部分情况下不会降低SFT的训练的效果。我们将4D混合并行策略与LoRA、P-Tuning等精调策略结合，可以在单机情况下(A100 80G*8)精调千亿模型，并显著升训练吞吐；
               

<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/3d25e8e7-77fe-49d7-b4df-ed51e61f56b0" alt="PEFT配置" width="35%" height="35%">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/07135425-5e30-4fec-80da-c005d9133df7" alt="性能对比" width="33.5%" height="33.5%">
</p>

                
### 大模型压缩：自研Shift-SmoothQuant算法全面实现主流大模型无损量化

由于大模型预测成本高，对算力、显存要求高，同时在NLU模型相关上通过模型压缩量化可以做到无损量化，因此在NLG的生成模型做压缩量化有益于大模型的预测部署；PaddleFleetX 通过自研的Shift-SmoothQuant算法有效提升量化的精度和稳定性，通过 Shift 算法可以参数分布缩放到对称分布，同时通过 SmoothQuant 将异常参数值进行缩放合理范围内，因此通过 Shift-SmoothQuant 算法可以提升压缩的算法精度和稳定性；在 C-Eval 和 NL2SQL 两个数据集上在主流开源模型可以做到无损量化。更多技术介绍可以参考[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)

<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/2214d4eb-efe9-45b4-b540-32d9b9e10985" alt="PEFT配置" width="50%" height="50%">
   <img src="https://github.com/wawltor/PaddleFleetX/assets/16698950/ccbfafe4-0a5d-472b-ad88-e844a1b44468" alt="PEFT配置" width="50%" height="50%">

</p>

              
### 大模型推理部署：高性能推理引擎与推理服务的深度结合

在PaddleFleetX通过Paddle Inference高性能推理引擎针对大模型Context与Decoder阶段的计算特性的不同，实现了大量的算子的融合与加速策略，结合动态插入技术，可以进一步提升推理服务的吞吐。
<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleFleetX/assets/16698950/10872ada-a629-473c-bf17-e01192165e4d" alt="PEFT配置" width="70%" height="70%">
</p>



## PaddleFleetX 应用案例

### 大语言模型

基于PaddleFleetX的核心能力，我们在PaddleNLP中提供了丰富的大语言模型全流程开发，更多详细使用说明可以参考[PaddleNLP LLM](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm)

### 跨模态大模型

除了大语言模型外，PaddleFleetX还包含了跨模态大模型的开发与训练，包括基于扩散模型的文生图、图生文等经典能力，更多详细使用说明可以参考[PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX)

### 生物计算大模型

在生物计算领域，基于PaddleFleetX的并行策略与高性能优化能力，我们在PaddleHelix中提供众多业界领先的生物计算预训练模型，更多详细使用说明可以参考[PaddleHelix](https://github.com/PaddlePaddle/PaddleHelix)


## Citation

```
@misc{paddlefleetx,
    title={PaddleFleetX: An Easy-to-use and High-Performance One-stop Tool for Deep Learning},
    author={PaddleFleetX Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleFleetX}},
    year={2022}
}
```

## License
PaddleFleetX 基于 [Apache 2.0 license](./LICENSE) 许可发布。