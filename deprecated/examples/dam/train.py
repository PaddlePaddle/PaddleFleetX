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
"""
Deep Attention Matching Network
"""
import sys
import os
import six
import numpy as np
import time
import multiprocessing
import paddle
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import evaluation as eva
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy

try:
    import cPickle as pickle  #python 2
except ImportError as e:
    import pickle  #python 3

import utils
import config
import model

def test():
    args = config.parse_args()
    config.print_arguments(args)

    trainer = GPUTrainer(distributed=False)
    place = trainer.place
    exe = trainer.exe
    dam = model.DAM(exe, args.max_turn_num, args.max_turn_len,
              args.vocab_size, args.emb_size, args.stack_num,
              args.channel1_num, args.channel2_num)

    filelists = {
        "test": [args.test_data_path],
    }
    loaders = dam.get_loader_from_filelist(
            args.batch_size, args.vocab_path, 
            args.data_source, filelists, 1, 0)
    test_loader = loaders["test"]

    trainer.run(dam, model_path=args.model_path)
    trainer.test(args, dam, test_loader)


def train():
    args = config.parse_args()
    config.print_arguments(args)

    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)

    trainer = GPUTrainer()
    place = trainer.place
    exe = trainer.exe
    dam = model.DAM(exe, args.max_turn_num, args.max_turn_len,
              args.vocab_size, args.emb_size, args.stack_num,
              args.channel1_num, args.channel2_num)

    filelists = {
        "train": [args.train_data_path],
        "valid": [args.valid_data_path],
    }
    loaders = dam.get_loader_from_filelist(
            args.batch_size, args.vocab_path, 
            args.data_source, filelists,
            fleet.worker_num(), fleet.worker_index())
    train_loader = loaders["train"]
    valid_loader = loaders["valid"]

    optimizer = fluid.optimizer.Adam(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=args.learning_rate,
            decay_steps=400,
            decay_rate=0.9,
            staircase=True),
        grad_clip=fluid.clip.GradientClipByValue(
            min=-1.0, max=1.0))
    
    optimizer = fleet.distributed_optimizer(optimizer)
    trainer.run(dam, optimizer=optimizer)

    if args.word_emb_init is not None:
        dam.init_emb_from_file(args.word_emb_init, place)

    epoch = args.num_scan_data
    trainer.fit(args, dam, train_loader, valid_loader, epoch)

class GPUTrainer(object):
    def __init__(self, distributed=True):
        self.distributed = distributed
        self.place_idx = int(os.environ['FLAGS_selected_gpus']) if self.distributed else 0
        self.place = fluid.CUDAPlace(self.place_idx)
        self.exe = fluid.Executor(self.place)

    def run(self, dam, optimizer=None, model_path=None):
        if model_path is not None:
            # test
            fluid.io.load_persistables(
                    executor=self.exe,
                    dirname=model_path,
                    main_program=fluid.default_main_program())
            self.test_prog = fluid.default_main_program()
        elif optimizer is not None:
            # train
            self.train_prog = fluid.default_main_program()
            self.start_prog = fluid.default_startup_program()
            optimizer.minimize(dam.loss, self.start_prog)
            self.test_prog = self.train_prog.clone(for_test=True)
            if self.distributed:
                self.train_prog = fleet.main_program
            self.exe.run(self.start_prog)

    def fit(self, args, dam, train_loader, valid_loader, epoch):
        if not os.path.exists(args.save_path):
            if fleet.worker_index() == 0:
                os.makedirs(args.save_path)

        train_fetch = [dam.loss]
        valid_fetch = [dam.logits] + ["label"]

        print_step = max(1, 1000000 / args.batch_size // (4 * 100))
        save_step = max(1, 1000000 / args.batch_size // (4 * 10))
        step = 0
        for e_i in range(epoch):
            start = time.time()
            for idx, sample in enumerate(train_loader()):
                step += 1
                ret = self.exe.run(
                        program=self.train_prog,
                        feed=sample,
                        fetch_list=train_fetch)
                if step % print_step == 0:
                    print('[TRAIN] epoch=%d step=%d loss=%f' % (e_i, step, ret[0][0]))
                if step % save_step == 0:
                    save_path = os.path.join(args.save_path, 
                            "model.epoch_{}.step_{}".format(e_i, step))
                    if fleet.worker_index() == 0:
                        fleet.save_persistables(executor=self.exe, dirname=save_path)
                        print("model saved in {}".format(save_path))
                    
                    filename = os.path.join(args.save_path,
                            "score.epoch_{}.step_{}.worker_{}".format(
                                e_i, step, fleet.worker_index()))
                    score_file = open(filename, "w")
                    for idx, sample in enumerate(valid_loader()):
                        ret = self.exe.run(
                                program=self.test_prog,
                                feed=sample,
                                fetch_list=valid_fetch,
                                return_numpy=False)
                        scores = np.array(ret[0])
                        label = np.array(ret[1])
                        for i in six.moves.xrange(len(scores)):
                            score_file.write("{}\t{}\n".format(scores[i][0], int(label[i][0])))
                    score_file.close()
                
                    if args.ext_eval:
                        result = eva.evaluate_douban(filename)
                        dist_result = utils.dist_eval_douban(self.exe, result)
                    else:
                        result = eva.evaluate_ubuntu(filename)
                        dist_result = utils.dist_eval_ubuntu(self.exe, result)

                    print("[VALID] global result: ")
                    for metric in dist_result:
                        print("[VALID]   {}: {}".format(metric, dist_result[metric]))
                    if fleet.worker_index() == 0:
                        with open(os.path.join(args.save_path,
                            "result.epoch_{}.step_{}".format(e_i, step)), "w") as f:
                            for k, v in dist_result.items():
                                f.write("{}: {}".format(k, v) + "\n")
            end = time.time()
            print("train epoch {} time: {} s".format(e_i, end - start))

    def test(self, args, dam, loader):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        start = time.time()
        filename = os.path.join(args.save_path, "score.test")
        score_file = open(filename, "w")
        fetch = [dam.logits] + ["label"]
        for idx, sample in enumerate(loader()):
            ret = self.exe.run(
                    program=self.test_prog,
                    feed=sample,
                    fetch_list=fetch,
                    return_numpy=False)
            scores = np.array(ret[0])
            label = np.array(ret[1])
            for i in six.moves.xrange(len(scores)):
                score_file.write("{}\t{}\n".format(scores[i][0], int(label[i][0])))
        score_file.close()
        
        if args.ext_eval:
            result = eva.evaluate_douban(filename)
        else:
            result = eva.evaluate_ubuntu(filename)

        print("[TEST] result: ")
        with open(os.path.join(args.save_path, "result"), "w") as f:
            for metric in result:
                value, length = result[metric]
                print("[TEST]   {}: {}".format(metric, 1.0 * value / length))
                f.write("{}: {}".format(metric, 1.0 * value / length) + "\n")
        end = time.time()
        print("test time: {} s".format(end - start))

if __name__ == '__main__':
    args = config.parse_args()
    if args.do_train:
        train()
    if args.do_test:
        test()
