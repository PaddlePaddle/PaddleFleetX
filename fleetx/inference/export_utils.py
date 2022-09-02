# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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

import os
import paddle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['export_inference_model']


def _prune_input_spec(input_spec, program, targets):
    # try to prune static program to figure out pruned input spec
    # so we perform following operations in static mode
    device = paddle.get_device()
    paddle.enable_static()
    paddle.set_device(device)
    pruned_input_spec = []
    program = program.clone()
    program = program._prune(targets=targets)
    global_block = program.global_block()
    for spec in input_spec:
        try:
            v = global_block.var(spec.name)
            pruned_input_spec.append(spec)
        except Exception:
            pass
    paddle.disable_static(place=device)
    return pruned_input_spec


def export_inference_model(model, input_spec, save_dir='./output'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    static_model = paddle.jit.to_static(model, input_spec)
    pruned_input_spec = _prune_input_spec(input_spec,
                                          static_model.forward.main_program,
                                          static_model.forward.outputs)
    paddle.jit.save(static_model, save_dir, input_spec=pruned_input_spec)
    logger.info("export inference model saved in {}".format(save_dir))
