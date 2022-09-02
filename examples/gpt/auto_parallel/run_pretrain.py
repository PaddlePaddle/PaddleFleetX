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

import random
import sys
import numpy as np

import paddle
from paddle.static import InputSpec
from paddle.distributed import fleet

sys.path.append("../../../")
from examples.gpt.tools import parse_yaml_auto
from examples.gpt.gpt_module import GPTAutoModule
from fleetx.data.sampler import Stack, Tuple
from fleetx.core.engine import AutoEngine
from fleetx.data.tokenizers import GPTTokenizer
from fleetx.datasets.gpt import create_pretrained_dataset_auto, get_train_data_file


def generate_dist_strategy(configs):

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    dist_strategy.recompute = configs['Engine']['use_recompute']
    amp_configs = configs['Engine']['mix_precision']
    dist_strategy.amp = amp_configs['use_pure_fp16']
    dist_strategy.amp_configs = {
        "custom_black_list": amp_configs['custom_black_list'],
        "custom_white_list": amp_configs['custom_white_list'],
        "init_loss_scaling": amp_configs['scale_loss'],
        "use_pure_fp16": amp_configs['use_pure_fp16'],
    }
    sharding_configs = configs['Distributed']['sharding']
    dist_strategy.sharding = sharding_configs['sharding_degree'] > 1
    dist_strategy.sharding_configs = {
        "sharding_degree": sharding_configs['sharding_degree'],
        "stage": sharding_configs['sharding_stage']
    }
    return dist_strategy


def generate_data_holder(configs):
    data_configs = configs['Data']
    gbsz = data_configs['batch_size']['global_batch_size']
    max_seq_len = data_configs['dataset']['max_seq_len']

    tokens = InputSpec([gbsz, max_seq_len], "int64", "tokens")
    position_ids = InputSpec([gbsz, max_seq_len], "int64", "position_ids")
    labels = InputSpec([gbsz, max_seq_len], "int64", "labels")
    loss_mask = InputSpec([gbsz, max_seq_len], "float32", "loss_mask")

    return [tokens, position_ids], [labels, loss_mask]


def do_train():
    configs = parse_yaml_auto()
    seed = configs['Global']['seed']
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    fleet.init(is_collective=True)
    dist_strategy = generate_dist_strategy(configs)
    inputs_spec, labels_spec = generate_data_holder(configs)

    module = GPTAutoModule(configs)
    engine = AutoEngine(
        module=module,
        configs=configs,
        inputs_spec=inputs_spec,
        labels_spec=labels_spec,
        strategy=dist_strategy)
    engine.load()

    gbsz = configs['Data']['batch_size']['global_batch_size']
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    for epoch in range(configs['Engine']['num_train_epochs']):
        files = get_train_data_file(configs['Data']['dataset']['input_dir'])
        files.sort()
        num_files = len(files)
        for f_id in range(num_files):
            data_file = files[f_id]
            train_data, valid_data, test_data = create_pretrained_dataset_auto(
                configs, [data_file], tokenizer.eos_token_id)
            engine.fit(train_dataset=train_data,
                       batch_size=gbsz,
                       collate_fn=Tuple(Stack(), Stack(), Stack(), Stack()))

            # engine.evaluate(valid_dataset=valid_data, batch_size=gbsz, collate_fn=Tuple(Stack(), Stack(), Stack(), Stack()))
            # engine.predict(test_dataset=test_data, batch_size=gbsz, collate_fn=Tuple(Stack(), Stack(), Stack(), Stack()))
            # engine.save()


if __name__ == "__main__":
    do_train()
