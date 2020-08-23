#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from model import DAM
from trainer import GPUTrainer
import config
import paddle.distributed.fleet as fleet

def train():
    args= config.parse_args()
    config.print_arguments(args)

    fleet.init(is_collective=True)
    dam = DAM(args.max_turn_num, args.max_turn_len,
              args.vocab_size, args.emb_size,
              args.stack_num, args.channel1_num, args.channel2_num)
    train_loader, valid_loader = dam.get_loader_from_filelist(
        args.filelist, args.data_source, fleet.worker_num())
    dam.init_emb_from_file(args.word_emb_init)
    optimizer = paddle.optimizer.Adam(
        learning_rate=paddle.fluid.layers.exponential_decay(
            learning_rate=args.learning_rate,
            decay_steps=400,
            decay_rate-0.9,
            staircase=True),
        grad_clip=paddle.fluid.clip.GradientClipByValue(
            min=-1.0, max=1.0))
    optimizer = fleet.distributed_optimizer(optimizer)
    trainer = GPUTrainer()
    EPOCH = 5
    trainer.fit(dam, train_loader, valid_loader, optimizer, EPOCH)
    
if __name__ == '__main__':
    train()
