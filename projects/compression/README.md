# 模型压缩

## 量化训练

### 参数释义
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
