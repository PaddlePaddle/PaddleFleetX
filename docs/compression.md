# 模型压缩

------------------------------------------------------------------------------------------

## **简介**

PaddleFleetX 集成了 PaddleSlim 中的常见的压缩方法：量化训练（Qutization Aware Training，QAT）、结构化稀疏（Structured Pruning，SP）和知识蒸馏（Knowledge Distillation，KD）。本文会介绍如何在 PaddleFleetX 中使用这些功能，来压缩并且导出压缩后的模型。

## **特性**

- <a href=https://github.com/PaddlePaddle/PaddleSlim/tree/release/2.4/demo/dygraph/quant>量化训练</a>：通过将全连接层的矩阵乘计算由 Float 浮点型优化为 INT8 整型来优化推理性能；
- <a href=https://github.com/PaddlePaddle/PaddleSlim/tree/release/2.4/demo/dygraph/pruning>结构化稀疏</a>：通过剪裁全连接层权重的通道数目来优化推理性能；
- <a href=#知识蒸馏>知识蒸馏</a>：通过使用高精度的大模型（教师模型）来蒸馏低精度的小模型（学生模型）来提升小模型精度



## **配置文档**

模型压缩开关通过 Compress 字段控制，预训练的模型参数路径由 pretrained 指定。接下来就是量化训练、结构化稀疏和知识蒸馏各自的技术参数。

```yaml
Compress:
  pretrained:         // 预训练模型参数的保存路径
  
  Quantization:       // 量化训练参数
    
  Prune:              // 结构化稀疏参数
  
  Distillation:       // 知识蒸馏参数
```

**注意**： 我们正在开发上述三种压缩方法的联合使用，请先单独使用上述各个方法。

### **量化训练参数**

```yaml
Compress:
  pretrained:
  Quantization:
    enable: True
    weight_quantize_type: 'abs_max'
    activation_quantize_type: 'moving_average_abs_max'
    weight_preprocess_type: None
    activation_preprocess_type: 'PACT'
    weight_bits: 8
    activation_bits: 8
    quantizable_layer_type: ['Linear', 'ColumnParallelLinear', 'RowParallelLinear']
    onnx_format: True
```

其中参数说明：

| **参数名**                   | **参数释义**                              |
|-----------------------------|-----------------------------------------|
| pretrained                  | 预训练模型的加载目录，若设置该参数，将在量化之前加载预训练模型；若需要加载量化后参数，将此参数设置为None，直接设置Engine.save_load.ckpt_dir即可       |
| enable                      | 是否开启量化训练                           |
| weight_quantize_type        | weight量化方法, 默认为`channel_wise_abs_max`, 此外还支持`abs_max` |
| activation_quantize_type    | activation量化方法, 默认为`moving_average_abs_max`               |
| weight_preprocess_type      | weight预处理方法，默认为None，代表不进行预处理；当需要使用`PACT`方法时设置为`PACT` |
| activation_preprocess_type  | activation预处理方法，默认为None，代表不进行预处理                   |
| weight_bits                 | weight量化比特数, 默认为 8                                        |
| activation_bits             | activation量化比特数, 默认为 8                                    |
| quantizable_layer_type      | 需要量化的算子类型                                                |
| onnx_format                 | 是否使用新量化格式，默认为False                                     |

更详细的量化训练参数介绍可参考[PaddleSlim动态图量化训练接口介绍](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/dygraph/quanter/qat.rst)。

### **知识蒸馏参数**

```yaml
Compress:
  pretrained:
  Distillation:
    enable: True
    distill_loss_ratio: 0.1 
    distillation_loss_type: 'kl_div'
    T: 1.5
    Teacher:
      save_load:
        ckpt_dir: 
      Model:
        hidden_size: 2048
        vocab_size: 50304
        num_layers: 24
        num_attention_heads: 16 
        ffn_hidden_size: 8196
        hidden_dropout_prob: 0.1
        attention_probs_dropout_prob: 0.1
        max_position_embeddings: 1024
        type_vocab_size: 16
        initializer_range: 0.02
        use_recompute: False 
        recompute_granularity:
        no_recompute_layers:
```

其中参数对应的释义如下：

| **参数名**             | **参数释义**                                                |
| ---------------------- | ----------------------------------------------------------- |
| enable                 | 是否开启蒸馏                                                |
| distill_loss_ratio     | 蒸馏损失函数的权重                                          |
| distillation_loss_type | 蒸馏损失函数的类型，支持mse, kl_div和token_wise_contrastive |
| ckpt_dir               | 教师网络 checkpoint的加载目录                               |
| T                      | 学生网络除的温度                                            |

Model部分的参数意义和部分相同，如果不指定Model部分内容，则默认用学生网络用作教师网络。
