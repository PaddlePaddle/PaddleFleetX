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

from model_check import check_cuda
import utils
import config
import model

def main():
    args = config.parse_args()
    config.print_arguments(args)
    
    if args.do_train:
        if args.distributed:
            init_dist_env()

        place = create_place(args.distributed)
        exe = fluid.Executor(place)

        train_prog, start_prog = fluid.Program(), fluid.Program()
        with fluid.program_guard(train_prog, start_prog):
            feed, loss, optimizer = model.build_train_net(args)
            
        if args.distributed:
            optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(loss, start_prog)

        filelist = utils.load_filelist(args.filelist, args.distributed)
        print("files: {}".format(filelist))
            
        dataloader = utils.create_dataloader(
                feed_var_list=feed, 
                filelist=filelist,
                place=fluid.cpu_places(),
                batch_size=args.batch_size, 
                thread_num=4, 
                dict_path=args.vocab_path,
                max_turn_num=args.max_turn_num,
                max_turn_len=args.max_turn_len,
                is_test=False,
                data_source=args.data_source)
        
        if args.distributed:
            train_prog = fleet.main_program
            
        exe.run(start_prog)
        train(args, train_prog, exe, feed, [loss], dataloader)
    
    if args.do_test:
        assert args.distributed == False 

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if not os.path.exists(args.model_path):
            raise ValueError("Invalid model init path %s" % args.model_path)
        
        place = create_place(False)
        exe = fluid.Executor(place)

        model_path = args.model_path

        # create test network
        test_prog, start_prog = fluid.Program(), fluid.Program()
        with fluid.program_guard(test_prog, start_prog):
            feed, logits = model.build_test_net(args)
            fluid.io.load_persistables(
                    executor=exe,
                    dirname=model_path,
                    main_program=fluid.default_main_program())
        
        filelist = utils.load_filelist(args.filelist, False)
        print("files: {}".format(filelist))
        dataloader = utils.create_dataloader(
                feed_var_list=feed, 
                filelist=filelist,
                place=fluid.cpu_places(),
                batch_size=args.batch_size, 
                thread_num=4, 
                dict_path=args.vocab_path,
                max_turn_num=args.max_turn_num,
                max_turn_len=args.max_turn_len,
                is_test=True,
                data_source=args.data_source)

        test(args, test_prog, exe, feed, [logits], dataloader)

def init_dist_env():
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)

def create_place(is_distributed):
    place_idx = int(os.environ['FLAGS_selected_gpus']) if is_distributed else 0
    return fluid.CUDAPlace(place_idx)

def train(args, train_prog, exe, feed, fetch, loader):
    for epoch in range(args.num_scan_data):
        start = time.time()
        for idx, sample in enumerate(loader()):
            ret = exe.run(
                    program=train_prog,
                    feed=sample,
                    fetch_list=fetch)
            if idx % 10 == 0:
                print('[TRAIN] epoch=%d step=%d loss=%f' % (epoch, idx, ret[0][0]))
        end = time.time()
        print("epoch {}: {} s".format(epoch, end - start))

    save_path = os.path.join(args.save_path, "model.{}".format(epoch))
    if args.distributed and fleet.worker_index() == 0:
        fleet.save_persistables(executor=exe, dirname=save_path)
        print("model saved in {}".format(save_path))

def test(args, test_prog, exe, feed, fetch, loader):
    filename = os.path.join(args.save_path, "score")
    score_file = open(filename, 'w')
    fetch_list = fetch + ["label"]
    for idx, sample in enumerate(loader()):
        ret = exe.run(
                program=test_prog,
                feed=sample,
                fetch_list=fetch_list,
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
    for metric in result:
        print("[TEST]   {}: {}".format(metric, result[metric]))

if __name__ == '__main__':
    main()
