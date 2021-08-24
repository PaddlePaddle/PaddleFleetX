import paddle
import paddle.fluid.core as core
import paddle.distributed.fleet as fleet
import numpy as np
import random
import os

def fix_seed(seed=None): 
    if seed is None:
        seed = os.environ.get('FLAGS_paddle_seed', None)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        paddle.seed(seed)
        core.globals()['FLAGS_cudnn_deterministic'] = True


def update_strategy(strategy, cudnn_deterministic=True): 
    if not core.globals()['FLAGS_apply_pass_to_program']:
        return

    if core.globals()['FLAGS_max_inplace_grad_add'] <= 0:
        core.globals()['FLAGS_max_inplace_grad_add'] = 8

    settings = {
        "fuse_relu_depthwise_conv": True,
        "fuse_bn_act_ops": True,
        "fuse_bn_add_act_ops": True,
        "fuse_elewise_add_act_ops": True,
        "fuse_all_optimizer_ops": True,
        "enable_addto": True,
        "enable_inplace": True,
    }
    build_strategy = paddle.static.BuildStrategy() 
    for k, v in settings.items():
        setattr(build_strategy, k, v)
    strategy.build_strategy = build_strategy
    strategy.without_graph_optimization = True
