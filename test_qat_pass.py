# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle
import paddleslim

import ppfleetx.models.language_model.gpt as gpt
from ppfleetx.models.language_model.gpt.auto.auto_utils import process_mesh_config

mesh = process_mesh_config({'dp_degree': 1, 'mp_degree': 1, 'pp_degree': 1})
model_cfgs = {
    'vocab_size': 50304,
    'hidden_size': 1024,
    'num_layers': 4,
    'num_attention_heads': 4,
    'mesh': mesh
}
generation_cfgs = {
    'top_k': 50,
    'top_p': 0.75,
    'temperature': 1.0,
    'min_dec_len': 1,
    'max_dec_len': 200,
    'num_return_sequences': 1,
    'decode_strategy': 'sampling'
}

input_spec = [
    paddle.static.InputSpec(
        shape=[None, None], name="input_ids", dtype='int64')
]
with paddle.LazyGuard():
    model = gpt.GPTForGenerationAuto(
        gpt.GPTModelAuto(**model_cfgs), generation_cfgs)

model = paddle.jit.to_static(model, input_spec)
main_prog = model.forward.concrete_program.main_program
startup_prog = model._startup_program()

paddle.enable_static()

val_prog = main_prog.clone(for_test=True)
place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
) else paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(startup_prog)

config = {
    'weight_quantize_type': 'channel_wise_abs_max',
    'activation_quantize_type': 'moving_average_abs_max',
    'quantize_op_types': [
        'conv2d',
        'depthwise_conv2d',
        'mul',
        'matmul',
        'matmul_v2',
    ],
    'onnx_format': True
}
quant_eval_prog = paddleslim.quant.quant_aware(
    val_prog, place, config, for_test=True)
quant_eval_prog = paddleslim.quant.convert(quant_eval_prog, place, config)
print(quant_eval_prog)

paddle.fluid.io.save_inference_model(
    dirname='infer2',
    feeded_var_names=["input_ids"],
    target_vars=[quant_eval_prog.global_block().var('reshape2_17.tmp_0')],
    executor=exe,
    main_program=quant_eval_prog,
    model_filename='infer2/model',
    params_filename='infer2/params')
