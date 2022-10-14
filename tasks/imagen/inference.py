# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tqdm
import time
import random
import os
import sys
import cv2
import numpy as np

import argparse
import paddle
from paddle.distributed import fleet
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from paddle.jit import to_static
from paddle.static import InputSpec
from ppfleetx.utils import config, env
from ppfleetx.utils.log import logger
from ppfleetx.data import build_dataloader, tokenizers
from ppfleetx.models import build_module
from ppfleetx.core import EagerEngine

PATH = "/work/t5_project/imagen/models/imagen_text2im_397m_full"

def get_args():
    parser = argparse.ArgumentParser('T5 inference script', add_help=False)
    parser.add_argument("--save", action='store_true', default=False)
    parser.add_argument("--trt", action='store_true', default=False)
    parser.add_argument("--tune", action='store_true', default=False)
    parser.add_argument("--half", action='store_true', default=False)
    parser.add_argument("--infer", action='store_true', default=False)
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/config.yaml',
        help='config file path')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    return parser.parse_args()

def save_images(images, output='', num_unets=1):
    """ save images"""
    if not os.path.exists(output):
        os.makedirs(output)
    img_size = [64, 512]
    for i in range(num_unets):
        for ith, image in enumerate(images[i]):
            norm_image = cv2.normalize(
                image.transpose([1, 2, 0]).cpu().numpy(),
                None, alpha = 0, beta = 255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            norm_image = norm_image.astype(np.uint8)[..., ::-1]
            img_gen = output + '/' + 'display.paper_' + str(
                img_size[i])+ '.' + str(ith) +'.jpg'
            cv2.imwrite(img_gen, norm_image)

def export_model(model):
    from paddle.inference import convert_to_mixed_precision
    from paddle.inference import PrecisionType, BackendType
    x_spec = InputSpec(shape=[None, None], dtype='int64', name='x')
    y_spec = InputSpec(shape=[None, None], dtype='int64', name='y')
    static_model = to_static(model.sample, [x_spec, y_spec])
    for block in static_model.main_program.blocks:
        for op in block.ops:
            if op.has_attr("op_callstack"):
                op._remove_attr("op_callstack")
    model = paddle.jit.save(static_model, path=PATH)
    with open('/work/t5_project/imagen/models/imagen_text2im_397m_full.prog.txt', 'w') as f:
        f.write(str(static_model.main_program))

    black_list = {
        # "layer_norm",
        # "softmax",
        # "cast",
        # "elementwise_mul"
        }
    convert_to_mixed_precision(
                f'{PATH}.pdmodel',
                f'{PATH}.pdiparams',
                f'{PATH}_half.pdmodel',
                f'{PATH}_half.pdiparams',
                PrecisionType.Half, BackendType.GPU, True, black_list=black_list)
    print("Done")

def eval_model_performance(imagen_model):
    paddle.seed(100)
    np.random.seed(100)
    random.seed(100)

    input_ids = 128 * np.ones((1, 128)).astype(np.int64)
    text_masks = np.ones((1, 128)).astype(np.int64)

    start_time = time.perf_counter()
    for _ in tqdm.tqdm(range(10)):
        out = imagen_model.sample(paddle.to_tensor(input_ids), paddle.to_tensor(text_masks))
    end_time = time.perf_counter()
    print("eval duration: ", (end_time - start_time) / 10 * 1000, " ms")

def eval_model(imagen_model):
    paddle.seed(100)
    np.random.seed(100)
    random.seed(100)

    input_ids = 128 * np.ones((1, 128)).astype(np.int64)
    text_masks = np.ones((1, 128)).astype(np.int64)

    out = imagen_model.sample(paddle.to_tensor(input_ids), paddle.to_tensor(text_masks))
    print("eval: ", out[0].numpy().reshape(-1).tolist()[:20])

    x_spec = InputSpec(shape=[None, None], dtype='int64', name='input_ids')
    y_spec = InputSpec(shape=[None, None], dtype='int64', name='text_masks')

    sample_static = to_static(imagen_model.sample, [x_spec, y_spec])
    
    paddle.seed(100)
    np.random.seed(100)
    random.seed(100)

    out2 = sample_static(input_ids, text_masks)
    print("to static: ", out2[0].numpy().reshape(-1).tolist()[:20])

def infer_predictor():
    import paddle.inference as paddle_infer
    from paddle.inference import Config
    from paddle.inference import create_predictor

    Half = global_args.half
    Tune = global_args.tune
    TRT = global_args.trt
    FakeData = True

    prefix = ""
    if Half:
        prefix = "_half"
        paddle.set_default_dtype("float16")
    config = Config(f"{PATH}{prefix}.pdmodel", f"{PATH}{prefix}.pdiparams")
    
    config.enable_memory_optim()
    config.switch_ir_optim(True)
    config.ir_optim()
    config.enable_use_gpu(1000, 0)
    if TRT:
        config.enable_tensorrt_engine(
                        workspace_size=1 << 30,
                        precision_mode=paddle_infer.PrecisionType.Half,
                        # precision_mode=paddle_infer.PrecisionType.Float32,
                        # precision_mode=TRT_PRECISIONS[precision],
                        # max_batch_size=8,
                        max_batch_size=1,
                        min_subgraph_size=20,
                        use_static=False,
                        use_calib_mode=False)
    shape_range_info_filename = f"{PATH}_shape_info.pbtxt"
    if Tune:
        config.tensorrt_engine_enabled()
        config.collect_shape_range_info(shape_range_info_filename)

    if not Tune and shape_range_info_filename and \
                    os.path.exists(shape_range_info_filename):
        print("!!!!!enable_tuned_tensorrt_dynamic_shape")
        
        config.enable_tuned_tensorrt_dynamic_shape(
                shape_range_info_filename, False)

    all_pass = [
        "ir_params_sync_among_devices_pass"
    ]
    for pass_item in all_pass:
        config.delete_pass(pass_item)

    model = create_predictor(config)

    print("dddd input names: ", model.get_input_names())

    input_ids_tensor = model.get_input_handle(model.get_input_names()[0])
    text_masks_tensor = model.get_input_handle(model.get_input_names()[1])
    
    output_tensor = model.get_output_handle(model.get_output_names()[0])

    paddle.seed(100)
    np.random.seed(100)
    random.seed(100)

    # run model
    if FakeData:
        input_ids = 8 * np.ones((1, 8)).astype(np.int64)
        text_masks = np.ones((1, 8)).astype(np.int64)
    else:
        tokenizer = tokenizers.get_t5_tokenizer(name='/work/t5_project/t5/t5-3b')
        input_text = 'one bike.'
        input_texts = [input_text] * 4
        encoded = tokenizer.batch_encode_plus(input_texts)
        input_ids = encoded.input_ids
        text_masks = encoded.attention_mask

    input_ids_tensor.copy_from_cpu(input_ids)
    text_masks_tensor.copy_from_cpu(text_masks)

    start_time = time.perf_counter()
    for _ in tqdm.tqdm(range(10)):
        model.run()
        output = output_tensor.copy_to_cpu()
    end_time = time.perf_counter()

    print("predictor: ", output[0].reshape(-1).tolist()[:20])
    print("predictor duration: ", (end_time - start_time) / 10 * 1000, " ms")
    

if __name__ == "__main__":
    global_args = get_args()
    # args = config.parse_args()
    cfg = config.get_config(global_args.config, overrides=global_args.override, show=False)

    if dist.get_world_size() > 1:
        fleet.init(is_collective=True, strategy=env.init_dist_env(cfg))

    env.set_seed(cfg.Global.seed)
    module = build_module(cfg)
    config.print_config(cfg)
    

    tokenizer = tokenizers.get_t5_tokenizer(name='/work/t5_project/t5/t5-3b')
    engine = EagerEngine(configs=cfg, module=module, mode='inference')

    input_text = 'one bike.'
    input_texts = [input_text] * 4
    encoded = tokenizer.batch_encode_plus(input_texts)
    input_ids = encoded.input_ids
    attn_masks = encoded.attention_mask
    input_dict = {'input_ids': input_ids, 'text_masks': attn_masks}

    if global_args.save:
        export_model(engine._module.model)

    if global_args.eval:
        eval_model(engine._module.model)
        eval_model_performance(engine._module.model)
    if global_args.infer:
        infer_predictor()
    

    # img_outs = engine._module.forward(**input_dict)
    
    # save_images(img_outs, 'test', num_unets=2)

