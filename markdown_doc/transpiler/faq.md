# CPU分布式训练(Transplier)常见问题

- Q:分布式训练如何保存模型？
- 问题解答：
使用Fleet进行分布式训练，需要使用Fleet想用的模型保存接口进行保存。对应的接口主要有：`fleet.save_inference_model`、`fleet.save_persistables` 两个， `save_inference_model`主要是用于预测的，该API除了会保存预测时所需的模型参数，还会保存预测使用的模型结构。而`save_persistables` 会保存一个program中的所有运行相关的参数及中间状态，但是不保存该program对应的模型结构。 **分布式训练需要用 0 号 trainer进行模型保存， 相应的参数都会从Parameter Server端拉取后保存**。

------------

- Q: 分布式训练如何加载指定预训练参数？
- 问题解答：
TBD


