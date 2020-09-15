安装Paddle与FleetX
------------------

Paddle
~~~~~~

使用飞桨进行分布式训练的最小安装集合就是安装Paddle。从Paddle
2.0版本开始，我们面向不同用户群体提供不同类型的分布式训练API。

-  面向算法工程师为主的高级API **paddle.distributed.fleet**\ 。
-  面向具有分布式训练底层工程开发能力的工程师提供的API
   **paddle.distributed**\ 。
-  您只需要安装Paddle，就可以获得飞桨团队官方提供的所有分布式训练功能。

::

   pip install paddlepaddle-gpu

关于安装Paddle，\ `这里 <https://www.paddlepaddle.org.cn/install/quick>`__
有更完备的安装指南供您参考。

FleetX
~~~~~~

更大规模的数据、能够记忆并泛化大数据的模型、超大规模算力是利用深度学习技术提升业务效果的有效方法。我们针对需要大规模数据、大容量模型并需要进行高性能分布式训练的应用场景，开发了
**FleetX** 工具包。

-  在数据维度，提供用户可以快速定义的标准公开数据集以及低门槛替换自己业务数据集的接口。
-  在模型维度，FleetX提供典型的分布式训练场景下最常用的标准模型供用户直接使用，例如标准的预训练模型Resnet50、Ernie、Bert等。
-  在利用大规模算力集群方面，FleetX使用Paddle原生提供的分布式训练能力，面向不同的模型提供最佳的分布式训练实践，在保证收敛效果的前提下最大化用户的集群使用效率。

.. code:: bash

   pip install fleet-x
