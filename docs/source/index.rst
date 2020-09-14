:github_url: https://github.com/PaddlePaddle/Fleet

=========================================================
欢迎来到飞桨分布式技术文档主页
=========================================================

- Paddle Fleet是飞桨分布式训练的基础API，面向每个人提供高性能的分布式训练能力。
- FleetX是Paddle Fleet的一个扩展组件，提供分布式训练周边的各种工具，满足沉浸式分布式训练的使用需求！

.. toctree::
   :maxdepth: 1
   :caption: 1. 快速开始
   :name: new-users

   paddle_fleet_rst/install_cn
   paddle_fleet_rst/fleetrun_usage_cn
   paddle_fleet_rst/fleet_static_quick_start
   paddle_fleet_rst/fleet_dygraph_quick_start
   paddle_fleet_rst/fleetx_quick_start
   paddle_fleet_rst/fleetsub_quick_start

.. toctree::
   :maxdepth: 1
   :caption: 2. 数据并行训练
   :name: advanced-doc-data-parallel-training

   paddle_fleet_rst/fleet_ps_sync_and_async_cn
   paddle_fleet_rst/fleet_collective_training_practices_cn
   paddle_fleet_rst/fleet_collective_training_speedup_with_amp_cn
   paddle_fleet_rst/fleet_large_batch_training_techniques_cn
   paddle_fleet_rst/fleet_improve_large_batch_accuracy_cn.md

.. toctree::
   :maxdepth: 1
   :caption: 3: 模型并行与流水线并行
   :name: advanced-model-parallel-pipeline-parallel

   paddle_fleet_rst/fleet_model_parallel_cn
   paddle_fleet_rst/fleet_pipeline_parallel_cn

.. toctree::
   :maxdepth: 1
   :caption: 4: 云端训练实践
   :name: advanced-doc-model-parallel

   paddle_fleet_rst/fleet_ps_geo_async_cn	  
   paddle_fleet_rst/fleet_and_edl_for_distillation_cn
   paddle_fleet_rst/fleet_on_cloud_cn
   paddle_fleet_rst/fleet_local_sgd_for_large_batch_training

.. toctree::
   :maxdepth: 1
   :caption: 5. 飞桨分布式训练性能基准
   :name: benchmark

   paddle_fleet_rst/fleet_gpu_benchmark_cn
   paddle_fleet_rst/fleet_ps_benchmark_cn

.. toctree::
   :maxdepth: 1
   :caption: 6. 大规模场景应用案例
   :name: applications

   paddle_fleet_rst/fleet_from_training_to_serving_cn

.. toctree::
   :maxdepth: 1
   :caption: 7. 分布式训练FAQ

   paddle_fleet_rst/fleet_user_faq_cn

.. toctree::
   :maxdepth

=======

FleetX使用Apache License 2.0开源协议
