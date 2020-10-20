# fleet多策略组合并行与自动并行

## 简介
fleet中包含多种分布式优化策略。如可使用自动混合精度amp来提升训练速度，使用recompute来提高模型训练的batch size，亦或者使用lars策略来解决超大batch训练中的收敛问题。
在实际的并行分布式训练中，可以根据训练需求将多种策略组合起来。比如希望在超大batch size下尽快的训练模型收敛，则可以使用amp+recompute+lars的优化策略。
同时，我们在fleet中提供了auto自动并行的优化策略，支持分布式下多策略最优化自动并行。
## 多策略组合并行
我们以在单机多卡上训练Resnet50为例子简单介绍fleet中多策略组合并行的用法。下面介绍的代码将会开启amp+recompute+lars的多策略组合并行方案。
### 添加依赖
首先我们要导入依赖和定义模型和 data loader, 这一步和Fleet下其他任务基本一致.
``` python
import os
import fleetx as X
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import time
import paddle.distributed.fleet as fleet
```
### 定义分布式模式并初始化
``` python
paddle.enable_static()
configs = X.parse_train_configs()
fleet.init(is_collective=True)
```
### 加载模型及数据
``` python
model = X.applications.Resnet50()
downloader = X.utils.Downloader()
local_path = downloader.download_from_bos(
    fs_yaml='https://fleet.bj.bcebos.com/test/loader/small_imagenet.yaml',
    local_path='./data')
batch_size = 32
loader = model.get_train_dataloader(local_path, batch_size=batch_size)
```
### 定义分布式及相关策略组合
``` python
dist_strategy = fleet.DistributedStrategy()

dist_strategy.amp = True
dist_strategy.recompute = True
dist_strategy.recompute_configs = {"checkpoints": model.checkpoints}
dist_strategy.lars = True
dist_strategy.lars_configs = {
                    "lars_coeff": 0.001,
                    "lars_weight_decay": 0.0005,
                    "exclude_from_weight_decay": ['batch_norm', '.b_0']
                }

optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)
```
### 开始训练
这一部分和Fleet 中其他任务基本相同:
``` python
place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

for i, data in enumerate(loader()):
    start_time = time.time()
    cost_val = exe.run(model.main_prog,
                        feed=data,
                        fetch_list=[model.loss.name])

    end_time = time.time()
    print(
        "worker_index: %d, step%d cost = %f, speed: %f"
        % (fleet.worker_index(), i, cost_val[0], batch_size / (end_time - start_time)))
```

### 运行训练脚本
一行启动单机多卡分布式训练：
``` python
fleetrun --gpus 0,1,2,3,4,5,6,7 --log_dir log train.py

# worker_index: 0, step160 cost = 7.014869, speed: 327.341949
# worker_index: 0, step161 cost = 6.948522, speed: 333.044819
# worker_index: 0, step162 cost = 6.989668, speed: 322.420980
# worker_index: 0, step163 cost = 7.073019, speed: 313.191089
```

### 可组合的策略
目前fleet下，以下策略可正常组合运行。
| 编号 | 策略组合 |
| :-: | :-: |
| 1 | amp+recompute |
| 2 | dgc+recompute |
| 3 | lars+recompute |
| 4 | lamb + recompute |
| 5 | amp + localsgd |
| 6 | amp + adaptive_localsgd |
| 7 | amp + lars |
| 8 | amp + lamb |
| 9 | amp + recompute + lars |
| 10 | amp + recompute + lamb |
| 11 | ...待补充... |

若是开启的多种策略中存在不兼容的组合，则会根据策略组合中的最长路径选择策略。如开启amp+recompute+dgc+lars，dgc和amp尚不兼容，则最终生效组合为amp+recompute+lars。

## auto自动并行
auto自动并行会自动搜索可优化的最长路径，此功能目前是实验性功能。自动并行只有在用户只设置auto，不设置其它策略时才能生效。
### 实现原理
auto自动并行会尝试打开所有可打开的优化策略，以优化策略为点，策略间的可组合关系为边构建一张有向无环图，再使用最长路径搜索算法寻找到一条可优化的最长路径，以此实现自动并行。该功能还是实验性功能，尚有优化点，如不同模型环境下优化策略的开关及权重的配置。
### 定义分布式自动并行策略
仍以上述代码为例，只需修改分布式策略定义部分。
``` python
dist_strategy.auto = True

optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
optimizer.minimize(model.loss)
```

### 运行训练脚本
一行启动单机多卡分布式训练：
``` python
fleetrun --gpus 0,1,2,3,4,5,6,7 --log_dir log train.py

# worker_index: 0, step174 cost = 7.056047, speed: 339.836454
# worker_index: 0, step175 cost = 6.953920, speed: 343.564841
# worker_index: 0, step176 cost = 6.954030, speed: 343.746102
# worker_index: 0, step177 cost = 6.822068, speed: 336.930790
```