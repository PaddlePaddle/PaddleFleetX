from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import paddle

from paddle.distributed import fleet
import paddle.distributed as dist
from paddle.io import Dataset, BatchSampler, DataLoader

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from ppfleetx.utils import config
from ppfleetx.distributed.apis import env
from ppfleetx.data import build_dataloader
from ppfleetx.models import build_module
from ppfleetx.core import EagerEngine

np.random.seed(1)
random_data = np.random.randn(1, 3, 224, 224)
img = paddle.to_tensor(random_data).astype(np.float32)

def has_diff(dynamic_out, static_output, threshold):
    a = dynamic_out.flatten()
    b = static_output.flatten()
    for index in range(len(a)):
        diff = abs(a[index]-b[index])
        if(diff>threshold):
            print("diff:",a[index],b[index])
            return True
    return False



if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)
    env.set_seed(cfg.Global.seed)
    cfg.Engine.mix_precision.use_pure_fp16 = False
    module = build_module(cfg)
    engine = EagerEngine(configs=cfg, module=module, mode='eval')
    ## train_predict
    if cfg.Engine.save_load.ckpt_dir is not None:
        engine.load()
    engine._module.model.eval()
    dynamic_graph_output = engine._module.model.forward(img)
    ## inference
    cfg.Inference.TensorRT.precision='fp32'
    module = build_module(cfg)
    engine = EagerEngine(configs=cfg,module=module, mode='inference')
    static_graph_output = engine.inference([img])

    diff = has_diff(dynamic_graph_output.numpy(), static_graph_output['linear_99.tmp_1'], threshold=0.0001)

    assert diff == False, "For VIT model, dynamic_graph_output != static_graph_output"
    