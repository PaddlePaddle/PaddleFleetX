# 推理部署

模型训练完成后，可使用飞桨高性能推理引擎Paddle Inference通过如下方式进行推理部署。

## 1. 模型导出

以`ERNIE(345M)`模型为例


导出单卡`ERNIE(345M)`模型：
```bash
sh projects/ernie/auto_export_ernie_345M_mp1.sh
```

导出多卡`ERNIE(345M)`模型：
```bash
sh projects/ernie/auto_export_ernie_345M_mp2.sh
```

## 2. 推理部署

模型导出后，可通过`tasks/ernie/inference.py`脚本进行推理部署。

`ERNIE(345M)` 推理
```bash
bash projects/ernie/run_inference.sh
```

## 3. Benchmark

测试中
