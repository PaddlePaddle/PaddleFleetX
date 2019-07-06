#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""BERT pretraining."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import paddle
import paddle.fluid as fluid
from model.bert import BertModel, BertConfig
from train_args import parser
import paddle.fluid.profiler as profiler
from timer import BenchmarkTimer
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from reader.pre_sampled_pretraining import DataReader

args = parser.parse_args()

def create_model(pyreader_name, bert_config):
    pyreader = fluid.layers.py_reader(
        capacity=70,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, 1], [-1, 1],
                [-1, 1]],
        dtypes=[
            'int64', 'int64', 'int64',
            'float32', 'int64', 'int64', 'int64'
        ],
        lod_levels=[0, 0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, pos_ids, sent_ids,
     input_mask, mask_label,
     mask_pos, labels) = fluid.layers.read_file(pyreader)

    bert = BertModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=bert_config,
        weight_sharing=args.weight_sharing,
        use_fp16=args.use_fp16)

    next_sent_acc, mask_lm_loss, total_loss = bert.get_pretraining_output(
        mask_label, mask_pos, labels)


    return pyreader, next_sent_acc, mask_lm_loss, total_loss


def noam_decay(warmup_steps, learning_rate):
    scheduled_lr = fluid.layers.leanring_rate_scheduler.\
                   noam_decay(1 / (warmup_steps * (learning_rate ** 2)),
                              warmup_steps)
    return scheduled_lr

def linear_warmup_decay(warmup_steps, learning_rate, num_train_steps):
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")
        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                fluid.layers.tensor.assign(warmup_lr, lr)
            with switch.default():
                decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                    learning_rate=learning_rate,
                    decay_steps=num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
                fluid.layers.tensor.assign(decayed_lr, lr)
    return lr

def const_learning_rate(leanring_rate):
    scheduled_lr = fluid.layers.create_global_var(
        name=fluid.unique_name.generate("learning_rate"),
        shape=[1],
        value=learning_rate,
        dtype='float32',
        persistable=True)
    return scheduled_lr

def exclude_from_weight_decay(name):
    if name.find("layer_norm") > -1:
        return True
    bias_suffix = ["_bias", "_b", ".b_0"]
    for suffix in bias_suffix:
        if name.endswith(suffix):
            return True
        return False

def train(args):
    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()
    train_pyreader, next_sent_acc, mask_lm_loss, total_loss = create_model(
        pyreader_name='train_reader', bert_config=bert_config)

    if args.warmup_steps > 0:
        scheduled_lr = linear_warmup_decay(args.warmup_steps, args.learning_rate,
                                           args.num_train_steps)
    else:
        scheduled_lr = const_learning_rate(learning_rate)
    optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
    optimizer._learning_rate_map[fluid.default_main_program()] = scheduled_lr
    clip_norm_thres = 1.0

    # set stop gradient here
    param_list = dict()
    if args.weight_decay > 0:
        for param in fluid.default_main_program().global_block().all_parameters():
            param_list[param.name] = param * 1.0
            param_list[param.name].stop_gradient = True

    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    strategy = DistributedStrategy()
    optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
    opts, param_and_grads = optimizer.minimize(total_loss)

    if args.weight_decay > 0:
        for param, grad in param_and_grads:
            if exclude_from_weight_decay(param.name):
                continue
            with param.block.program._optimized_guard(
                    [param, grad]), fluid.framework.name_scope("weight_decay"):
                updated_param = param - param_list[
                    param.name] * args.weight_decay * scheduled_lr
                fluid.layers.assign(output=param, input=updated_param)
                
    place = fluid.CUDAPlace(int(os.getenv("FLAGS_selected_gpus")))
    dev_count = fluid.core.get_cuda_device_count()
                
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    
    data_reader = DataReader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        in_tokens=args.in_tokens,
        epoch=args.epoch,
        max_seq_len=args.max_seq_len)

    train_pyreader.decorate_tensor_provider(data_reader.data_generator())
    train_pyreader.start()
    steps = 0
    cost = [.0]
    lm_cost = [.0]
    acc = [.0]
    timer = BenchmarkTimer()
    timer_start_step = 20
    timer.set_start_step(timer_start_step)
    program_cache = True

    while steps < args.num_train_steps:
        steps += 1
        timer.time_begin()
        each_next_acc, each_mask_lm_cost, each_total_cost, np_lr = exe.run(
            fluid.default_main_program(),
            fetch_list=[next_sent_acc.name, mask_lm_loss.name,
                        total_loss.name, scheduled_lr.name
            ], use_program_cache=program_cache)
        timer.time_end()
        acc.extend(each_next_acc)
        lm_cost.extend(each_mask_lm_cost)
        cost.extend(each_total_cost)
        total_file, current_file_index, current_file = data_reader.get_progress()
        print("current learning_rate:%f" % np_lr[0])
        print("progress: %d/%d, step: %d, loss: %f, "
              "ppl: %f, next_sent_acc: %f, speed: %f steps/s, file: %s"
              % (current_file_index, total_file, steps,
                 np.mean(np.array(cost)),
                 np.mean(np.exp(np.array(lm_cost))),
                 np.mean(np.array(acc)),
                 timer.time_per_step(),
                 current_file))

if __name__ == '__main__':
    train(args)
