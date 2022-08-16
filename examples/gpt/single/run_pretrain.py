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

import argparse
import math
import os
import random
import time
import sys
import yaml
import numpy as np

import paddle
sys.path.append("../../../")
from examples.gpt.gpt_module import GPTModule
from examples.gpt.tools import parse_args, parse_yaml
from fleetx.datasets.gpt import create_pretrained_dataset, get_train_data_file
from fleetx.data.tokenizers import GPTTokenizer
from fleetx.core.engine.eager_engine import EagerEngine


def do_train():
    args, configs = parse_yaml(parse_args().config)
    paddle.set_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    tokenizer = GPTTokenizer.from_pretrained("gpt2")

    module = GPTModule(args)
    engine = EagerEngine(module=module, configs=configs)

    if args.ckpt_dir:
        engine.load(ckpt_dir=args.ckpt_dir)

    for epoch in range(args.num_train_epochs):
        files = get_train_data_file(args)
        files.sort()
        num_files = len(files)

        for f_id in range(num_files):
            data_file = files[f_id]
            train_data_loader, valid_data_loader, test_data_loader = create_pretrained_dataset(
                args, [data_file],
                local_rank=0,
                data_world_size=1,
                data_world_rank=0,
                max_seq_len=args.max_seq_len,
                eos_id=tokenizer.eos_token_id)
            # Bug fix, if not call valid_data_loader, the enumerate will call valid_data_loader
            # many times. and start a new random dataloader.
            valid_data_loader = valid_data_loader()
            test_data_loader = test_data_loader()

            engine.fit(train_data_loader=train_data_loader,
                       valid_data_loader=valid_data_loader,
                       epoch=epoch)

            # engine.evaluate(valid_data_loader=valid_data_loader, epoch=epoch)      
            # engine.predict(test_data_loader=test_data_loader, epoch=epoch)
            # engine.save()


if __name__ == "__main__":
    do_train()
