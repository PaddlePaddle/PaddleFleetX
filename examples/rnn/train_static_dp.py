# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial
import argparse
import os
import random

import numpy as np
import paddle
from data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from datasets import load_dataset
import paddle.static as static
import paddle.distributed.fleet as fleet

from model import RNNModel
from utils import convert_example

paddle.enable_static()
# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--dp_degree', type=int, default="1", help="The number of gpu cards for data parallel mode.")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate used to train.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./senta_word_dict.txt", help="The directory to dataset.")
parser.add_argument('--network', choices=['bow', 'lstm', 'bilstm', 'gru', 'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn'],
    default="rnn", help="Select which network to train, defaults to bilstm.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed=1000):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      dp_degree=1,
                      batchify_fn=None,
                      data_holders=None):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        batchify_fn(obj:`callable`, optional, defaults to `None`): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`).

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = False if mode == 'train' else False
    if mode == "train":
        device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_replicas=dp_degree, rank=device_id)
    else:
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=batchify_fn, return_list=False, feed_list=data_holders)
    return dataloader

def rnn_pretrain_forward(args, train_program, start_program, topo=None):
    with static.program_guard(train_program, start_program), paddle.utils.unique_name.guard():
        batch_size = args.batch_size
        tokens = static.data(name="tokens", shape=[batch_size, -1], dtype="int64")
        seq_len = static.data(name="ids", shape=[batch_size], dtype="int64")
        labels = static.data(name="labels", shape=[batch_size], dtype="int64")
        data_holders = [tokens, seq_len, labels]
        # Loads vocab.
        if not os.path.exists(args.vocab_path):
            raise RuntimeError('The vocab_path  can not be found in the path %s' %
                            args.vocab_path)

        vocab = Vocab.load_vocabulary(
            args.vocab_path, unk_token='[UNK]', pad_token='[PAD]')
        # Loads dataset.
        train_ds, dev_ds, test_ds = load_dataset(
            "chnsenticorp", splits=["train", "dev", "test"])

        # Constructs the newtork.
        vocab_size = len(vocab)
        num_classes = len(train_ds.label_list)
        pad_token_id = vocab.to_indices('[PAD]')

        model = RNNModel(
            vocab_size,
            num_classes,
            direction='forward',
            padding_idx=pad_token_id,
            pooling_type='max')
        

        # Reads data and generates mini-batches.
        tokenizer = JiebaTokenizer(vocab)
        trans_fn = partial(convert_example, tokenizer=tokenizer, is_test=False)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),  # input_ids
            Stack(dtype="int64"),  # seq len
            Stack(dtype="int64")  # label
        ): [data for data in fn(samples)]
        train_loader = create_dataloader(
            train_ds,
            trans_fn=trans_fn,
            batch_size=args.batch_size,
            mode='train',
            batchify_fn=batchify_fn, data_holders=data_holders, dp_degree=args.dp_degree)

        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(), learning_rate=args.lr)
        criterion = paddle.nn.CrossEntropyLoss()
        preds = model(tokens, seq_len)
        loss = criterion(preds, labels)

    return train_program, start_program, loss, train_loader, optimizer, data_holders



if __name__ == "__main__":
    paddle.set_device(args.device)
    set_seed()
    place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    train_program = static.Program()
    start_program = static.Program()
    train_program, start_program, loss, train_loader, optimizer, data_holders = \
        rnn_pretrain_forward(args, train_program, start_program)
    with paddle.static.program_guard(train_program, start_program), paddle.utils.unique_name.guard():
        strategy = fleet.DistributedStrategy()
        strategy.without_graph_optimization = False
        strategy.fuse_all_reduce_ops = False
        fleet.init(is_collective=True, strategy=strategy)
        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(loss)

    exe = paddle.static.Executor(place)
    exe.run(start_program)
    fetch = [loss]
    with open("train_program.txt", "w") as f:
        f.write(str(train_program))
    for i in range(10):
        for eval_step, batch in enumerate(train_loader):
            loss = exe.run(train_program, feed=batch, fetch_list=fetch)
            print("epoch: ", i, "step: ", eval_step, " loss: ", loss[0])

   