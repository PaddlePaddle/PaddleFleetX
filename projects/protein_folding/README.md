# Protein Folding

声明: 本项目不提供具体能运行的蛋白质结构预测程序，如果想体验直接能运行的蛋白质结构预测代码，请跳转到
[HelixFold](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_folding/helixfold) 中运行。


本项目是一个教程，展示如何将数据并行、动态轴并行、分支并行（DP-DAP-BP）混合并行接入到 HelixFold 中。
想要在 HelixFold 中使用混合并行，则涉及到以下几个方面：

* 依赖安装
* 通信初始化
* 混合并行网络模型使用
* 优化器设置 DAP 和 BP 属性
* 参数同步与梯度同步

## 依赖安装
```shell
pip install ppfleetx
```

## 通信初始化

```python
from ppfleetx.distributed.protein_folding import dp
from ppfleetx.distributed.protein_folding.scg import scg

def init_distributed_env(args):
    dp_rank = 0 # ID for current device in distributed data parallel collective communication group
    dp_nranks = 1 # The number of devices in distributed data parallel collective communication group
    if args.distributed:
        # init bp, dap, dp hybrid distributed environment
        scg.init_process_group(parallel_degree=[('dp', None), ('dap', args.dap_degree), ('bp', args.bp_degree)])

        dp_nranks = dp.get_world_size()
        dp_rank = dp.get_rank_in_group() if dp_nranks > 1 else 0

        if args.bp_degree > 1 or args.dap_degree > 1:
            assert args.seed is not None, "BP and DAP should be set seed!"

    return dp_rank, dp_nranks
```

## 混合并行网络模型使用

目前，在 HelixFold 网络模型中涉及到混合并行的有 Embedding 和 Evoformer 类，因此可以将原来 HelixFold 中的 `EmbeddingsAndEvoformer`
修改为 `DistEmbeddingsAndEvoformer`。在网络模型中涉及 `DAP` 和 `BP` 的网络模型修改都在 [DistEmbeddingsAndEvoformer](../../ppfleetx/models/protein_folding/evoformer.py) 中封装，

```python
from ppfleetx.models.protein_folding.evoformer import DistEmbeddingsAndEvoformer 
evoformer = DistEmbeddingsAndEvoformer(
    self.channel_num, self.config.embeddings_and_evoformer,
    self.global_config)
```

## 优化器设置 DAP 和 BP 属性

由于 `DAP` 和 `BP` 在网络模型中分别切分的是中间激活值和网络计算分支，参数是没有切分的，因此在梯度同步的时候，
是需要区分同步的。我们将 `dap` 和 `bp` 属性设置在优化器参数分组中作为区分，并在后续梯度同步的时候使用。

```python
evoformer_params = []
template_and_pair_transition_params = []
other_params = []
for name, p in model.named_parameters():
    if 'template_pair_stack' in name or 'pair_transition' in name:
        template_and_pair_transition_params.append(p)
    elif 'evoformer_iteration' in name or 'extra_msa_stack' in name:
        evoformer_params.append(p)
    else:
        other_params.append(p)
parameters = []

if args.dap_degree > 1 or args.bp_degree > 1:
    parameters.append({'params': get_fused_params(other_params)})
    parameters.append({'params': get_fused_params(evoformer_params), 'dap': True, 'bp': True})
    parameters.append({'params': get_fused_params(template_and_pair_transition_params), 'dap': True})
else:
    parameters.append({'params': get_fused_params(other_params + evoformer_params + template_and_pair_transition_params)})

optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler, 
        epsilon=1e-06,
        grad_clip=grad_clip,
        parameters = parameters
    )
```

## 参数同步与梯度同步

### 参数同步

虽然是 `DP-DAP-BP` 混合并行，但是每个设备上的模型参数是没有切分的，因为在模型训练之前也需要做一次参数同步。

```python
from ppfleetx.distributed.protein_folding import dp

model = RunModel(train_config, model_config)
dp.param_sync(model, src_rank=0)
```

### 梯度同步

如上节所述，在梯度同步的时候需要分别对 `DP`，`DAP`，`BP` 并行策略相关的模型参数的梯度进行同步。

```python
from ppfleetx.distributed.protein_folding import dap, bp, dp

loss.backward()

# sync the gradient for branch parallel firstly
bp.grad_sync(optimizer._param_groups)
# then sync the gradient for dap
dap.grad_sync(optimizer._param_groups)
# finally sync the gradient for ddp
dp.grad_sync(optimizer._param_groups)

optimizer.step()
optimizer.clear_grad()
```

## 论文引用
```
@article{wang2022helixfold,
  title={HelixFold: An Efficient Implementation of AlphaFold2 using PaddlePaddle},
  author={Wang, Guoxia and Fang, Xiaomin and Wu, Zhihua and Liu, Yiqun and Xue, Yang and Xiang, Yingfei and Yu, Dianhai and Wang, Fan and Ma, Yanjun},
  journal={arXiv preprint arXiv:2207.05477},
  year={2022}
}

@article{wang2022efficient_alphafold2,
  title={Efficient AlphaFold2 Training using Parallel Evoformer and Branch Parallelism},
  author={Wang, Guoxia and Wu, Zhihua and Fang, Xiaomin and Xiang, Yingfei and Liu, Yiqun and Yu, Dianhai and Ma, Yanjun},
  journal={arXiv preprint arXiv:2211.00235},
  year={2022}
}
```
