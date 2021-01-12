分布式指标
=====================

简介
-------------------
分布式指标是指在分布式训练任务中用以评测模型效果的指标。它和单机模型评测指标的区别在于，单机指标仅评测当前节点的测试数据，而分布式指标需评测所有节点的全量测试数据。

原理
------

分布式指标计算一般包含三步，下面我们以分布式准确率为例介绍整个过程。

1. 初始化分布式训练环境

    .. code:: python
        
        import paddle.distributed.fleet as fleet
        fleet.init()

2. 定义指标计算需要的所有中间状态统计值，每个训练节点统计各自的状态值。准确率计算需要样本总数和正确分类的样本数两个统计值。

    .. code:: python
        
        ...
        predict, label = model()

        # 1. 定义中间状态统计值，样本总数和正确分类的样本数
        correct_cnt = fluid.layers.create_global_var(name="right_cnt", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = fluid.layers.create_global_var(name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)
        
        # 2. 训练节点自己的状态统计
        batch_cnt = paddle.reduce_sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_accuracy = paddle.metric.accuracy(input=predict, label=label)
        batch_correct = batch_cnt * batch_accuracy

        total_cnt = total_cnt + batch_cnt
        correct_cnt = correct_cnt + batch_correct

3. 所有训练节点间进行 `all_reduce` 操作，获取全局统计值，然后根据指标计算公式，计算全局指标。

    .. code:: python
        
        global_cnt = fleet.metrics.sum(total_cnt)
        global_correct = fleet.metrics.accuracy(corrent_cnt, total_cnt)
        global_accuracy = global_correct / global_cnt


使用方法
-------------------
为方便使用，Paddle在 `paddle.distributed.metrics` 下将常见的一些指标计算进行了封装，下面对这些API的功能及参数进行说明，并提供用法示例。

分布式AUC
~~~~~~~~~~~~~~

.. py:function:: paddle.distributed.fleet.metrics.auc(stat_pos, stat_neg, scope=None, util=None)

[AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)是一个二分类任务中常用的效果评价指标，指ROC曲线和横坐标轴之间的面积，该值介于0～1之间，越大代表分类器效果越好。

**参数：**

    - stat_pos, (numpy.array|Tensor|string, required): 单机正样例中间统计结果，即单机 `paddle.metrics.auc` 的 `stat_pos` 输出。
    - stat_neg, (numpy.array|Tensor|string, required): 单机负样例中间统计结果，即单机 `paddle.metrics.auc` 的 `stat_neg` 输出。
    - scope, (Scope, optional)，作用域，若为None，则使用全局/默认作用域，默认为None。
    - util, (UtilBase, optinal)，分布式训练工具类，若为None，则使用默认工具类 `fleet.util`， 默认作用域，默认为None。

**用法示例：**

    .. code:: python
        
        ...
        predict, label = model()

        # 1. 单机组网阶段，计算正负样例中间统计结果。
        auc, batch_auc, [stat_pos, stat_neg, batch_stat_pos, batch_stat_neg] = \
            paddle.metrics.auc(input=predict, label=label)

        # 2. 分布式训练阶段，计算全局AUC。
        global_auc = fleet.metrics.auc(stat_pos, stat_neg)


分布式Accuracy
~~~~~~~~~~~~~~

.. py:function:: paddle.distributed.fleet.metrics.acc(correct, total, scope=None, util=None)
    
分布式准确率。准确率（Accuracy)是分类任务中常用的一个效果评价指标。通过比对预测标签和实际标签是否一致，从而计算模型的分类效果，公式如下：
    
    .. math::
        accuarcy &= \frac{right\_cnt}{total\_cnt}

其中，`right_cnt` 是预测标签等于真实标签的样本总数，`total_cnt` 是全部样本总数。


**参数：**

    - correct, (numpy.array|Tensor|string, required): 单机预测标签等于真实标签的样本总数。
    - total, (numpy.array|Tensor|string, required): 单机样本总数。
    - scope, (Scope, optional)，作用域，若为None，则使用全局/默认作用域，默认为None。
    - util, (UtilBase, optinal)，分布式训练工具类，若为None，则使用默认工具类 `fleet.util`， 默认作用域，默认为None。

**用法示例：**

    .. code:: python
        
        ...
        predict, label = model()

        # 1. 单机组网阶段，计算样本总数和预测正确的样本数
        correct_cnt = fluid.layers.create_global_var(name="right_cnt", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = fluid.layers.create_global_var(name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)
        
        batch_cnt = paddle.reduce_sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_accuracy = paddle.metrics.accuracy(input=predict, label=label)
        batch_correct = batch_cnt * batch_accuracy

        correct_cnt = correct_cnt + batch_correct
        total_cnt = total_cnt + batch_cnt

        # 2. 分布式训练阶段，计算全局准确率。
        global_accuracy = fleet.metrics.accuarcy(correct_cnt, total_cnt) 


分布式MAE
~~~~~~~~~~~~~~

.. py:function:: paddle.distributed.fleet.metrics.mae(abserr, ins_num, scope=None, util=None)

平均绝对误差(Mean Absolute Error)。平均绝对误差是绝对误差的平均值，一般用于计算 `loss` 损失值。
    
    .. math::

        abserr &= \sum |input - label|

        mae &= \frac{abserr}{ins\_num}

其中，`input` 是样本预测结果， `label` 是样本真实标签，`ins_num` 是样本总数，`abserr` 为绝对误差和。

**参数： **

    - abserr, (numpy.array|Tensor|string, required): 单机绝对误差和统计值。
    - ins_num, (numpy.array|Tensor|string, required): 单机样本总数。
    - scope, (Scope, optional)，作用域，若为None，则使用全局/默认作用域，默认为None。
    - util, (UtilBase, optinal)，分布式训练工具类，若为None，则使用默认工具类 `fleet.util`， 默认作用域，默认为None。


**用法示例：**

    .. code:: python
        
        ...
        predict, label = model()

        # 1. 单机组网阶段，计算绝对误差和样本总数
        abserr = fluid.layers.create_global_var(name="abserr", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = fluid.layers.create_global_var(name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)
        
        batch_cnt = paddle.reduce_sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_abserr = paddle.nn.functional.l1_loss(input, label, reduction='sum')

        total_cnt = total_cnt + batch_cnt
        abserr = abserr + batch_abserr

        # 2. 分布式训练阶段，计算全局准确率。
        global_mae = fleet.metrics.mae(abserr, total_cnt) 


分布式MSE
~~~~~~~~~~~~~~

.. py:function:: paddle.distributed.fleet.metrics.mse(sqrerr, ins_num, scope=None, util=None)

均方误差(Mean Squared Error)。一般用于计算 `loss` 损失值。
    
    .. math::

        sqrerr &= \sum (input - label)^2
        
        mse &= \frac{sqrerr}{ins\_num}

其中，`input` 是样本预测结果， `label` 是样本真实标签，`ins_num` 是样本总数， `sqrerr` 为平方误差和，

**参数：**

    - sqrerr, (numpy.array|Tensor|string, required): 单机平方误差和统计值。
    - ins_num, (numpy.array|Tensor|string, required): 单机样本总数。
    - scope, (Scope, optional)，作用域，若为None，则使用全局/默认作用域，默认为None。
    - util, (UtilBase, optinal)，分布式训练工具类，若为None，则使用默认工具类 `fleet.util`， 默认作用域，默认为None。

**用法示例：**

.. code:: python
    
    ...
    predict, label = model()

    # 1. 单机组网阶段，计算平方误差和样本总数
    sqrerr = fluid.layers.create_global_var(name="sqrerr", persistable=True, dtype='float32', shape=[1], value=0)
    total_cnt = fluid.layers.create_global_var(name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)
    
    batch_cnt = paddle.reduce_sum(
        paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
    batch_sqrerr = paddle.nn.functional.mse_loss(input, label, reduction='sum')

    total_cnt = total_cnt + batch_cnt
    sqrerr = sqrerr + batch_sqrerr

    # 2. 分布式训练阶段，计算全局准确率。
    global_mse = fleet.metrics.mse(sqrerr, total_cnt) 

分布式RMSE
~~~~~~~~~~~~~~

.. py:function:: paddle.distributed.fleet.metrics.rmse(sqrerr, total_ins_num, scope=None, util=None)

RMSE，均方根误差(Root Mean Squared Error)，是均方误差的算术平方根，亦称标准误差，一般用于计算 `loss` 损失值。
    
    .. math::

        sqrerr &= \sum (input - label)^2
        
        rmse &= \sqrt{\frac{sqrerr}{ins\_num}}

其中，`input` 是样本预测结果， `label` 是样本真实标签，`ins_num` 是样本总数， `sqrerr` 为平方误差和，

**参数：**

    - sqrerr, (numpy.array|Tensor|string, required): 单机平方误差和统计值。
    - ins_num, (numpy.array|Tensor|string, required): 单机样本总数。
    - scope, (Scope, optional)，作用域，若为None，则使用全局/默认作用域，默认为None。
    - util, (UtilBase, optinal)，分布式训练工具类，若为None，则使用默认工具类 `fleet.util`， 默认作用域，默认为None。


**用法示例：**

    .. code:: python
        
        ...
        predict, label = model()

        # 1. 单机组网阶段，计算平方误差和样本总数
        sqrerr = fluid.layers.create_global_var(name="sqrerr", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = fluid.layers.create_global_var(name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)
        
        batch_cnt = paddle.reduce_sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_sqrerr = paddle.nn.functional.mse_loss(input, label, reduction='sum')

        total_cnt = total_cnt + batch_cnt
        sqrerr = sqrerr + batch_sqrerr

        # 2. 分布式训练阶段，计算全局准确率。
        global_rmse = fleet.metrics.rmse(sqrerr, total_cnt) 

分布式Sum
~~~~~~~~~~~~~~

.. py:function:: paddle.distributed.fleet.metrics.sum(input, scope=None, util=None)

分布式求和。一般用于自定义指标计算。

**参数：**

    - input, (numpy.array|Tensor|string, required)，需要分布式求和的输入参数。
    - scope, (Scope, optional)，作用域，若为None，则使用全局/默认作用域，默认为None。
    - util, (UtilBase, optinal)，分布式训练工具类，若为None，则使用默认工具类 `fleet.util`， 默认为None。

**用法示例：**

    .. code:: python
        
        ...
        # 1. 单机组网阶段，计算Loss
        loss = model()

        # 2. 分布式训练阶段，计算全局Loss和
        total_loss = fleet.metrics.sum(loss) 

分布式Max
~~~~~~~~~~~~~~

.. py:function:: paddle.distributed.fleet.metrics.max(input, scope=None, util=None)

分布式求最大值。一般用于自定义指标计算。

**参数：**

    - input, (numpy.array|Tensor|string, required)，需要分布式求最大值的输入参数。
    - scope, (Scope, optional)，作用域，若为None，则使用全局/默认作用域，默认为None。
    - util, (UtilBase, optinal)，分布式训练工具类，若为None，则使用默认工具类 `fleet.util`， 默认为None。

**用法示例：**

    .. code:: python
        
        ...
        # 1. 单机组网阶段，计算Loss
        loss = model()

        # 2. 分布式训练阶段，计算全局最大Loss
        total_loss = fleet.metrics.max(loss)

分布式Min
~~~~~~~~~~~~~~

.. py:function:: paddle.distributed.fleet.metrics.min(input, scope=None, util=None)

分布式求最小值。一般用于自定义指标计算。

**参数：**

    - input, (numpy.array|Tensor|string, required)，需要分布式求最大值的输入参数。
    - scope, (Scope, optional)，作用域，若为None，则使用全局/默认作用域，默认为None。
    - util, (UtilBase, optinal)，分布式训练工具类，若为None，则使用默认工具类 `fleet.util`， 默认作用域，默认为None。

**用法示例：**

    .. code:: python
        
        ...
        # 1. 单机组网阶段，计算Loss
        loss = model()

        # 2. 分布式训练阶段，计算全局最小Loss
        total_loss = fleet.metrics.min(loss)
