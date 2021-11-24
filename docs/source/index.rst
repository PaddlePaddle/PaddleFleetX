:github_url: https://github.com/PaddlePaddle/FleetX

=========================================================
欢迎来到飞桨分布式技术文档主页
=========================================================

欢迎您关注飞桨分布式训练，我们希望能帮助每一位用户走上大规模工业化生产之路！

飞桨分布式训练技术源自百度的业务实践，在自然语言处理、计算机视觉、搜索和推荐等领域经过超大规模业务的检验。飞桨分布式支持参数服务器和基于规约(Reduce)模式的两种主流分布式训练构架，具备包括数据并行、模型并行和流水线并行等在内的完备的并行能力，提供简单易用地分布式训练接口和丰富的底层通信原语。本文档旨在帮助用户快速了解如何使用飞桨分布式，详解各种飞桨分布式能力和使用方法，赋能用户业务发展。

.. toctree::
   :maxdepth: 1 
   :caption: 飞桨分布式概览
   :name: distributed_training
   
   paddle_fleet_rst/distributed_introduction

.. toctree::
   :maxdepth: 2 
   :caption: 使用指南
   :name: distributed_training
   
   paddle_fleet_rst/install_cn
   paddle_fleet_rst/collective/index
   paddle_fleet_rst/parameter_server/index
   paddle_fleet_rst/launch


.. toctree::
   :maxdepth: 1
   :caption: 高阶内容
   :name: higher

   paddle_fleet_rst/distill
   paddle_fleet_rst/edl

.. toctree::
   :maxdepth: 1
   :caption: 分布式训练搭建方案
   :name: install

   paddle_fleet_rst/public_cloud
   paddle_fleet_rst/paddle_on_k8s

.. toctree::
   :maxdepth: 1
   :caption: 附录
   :name: appendix

   paddle_fleet_rst/benchmark
   paddle_fleet_rst/faq

=======

FleetX使用Apache License 2.0开源协议
