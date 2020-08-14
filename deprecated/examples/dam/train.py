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

        if not os.path.exists(args.save_path):
            if not args.distributed or fleet.worker_index() == 0:
                os.makedirs(args.save_path)
    
        place = create_place(args.distributed)
        exe = fluid.Executor(place)

        train_prog, start_prog = fluid.Program(), fluid.Program()
        with fluid.program_guard(train_prog, start_prog):
            dam, feed, loss, optimizer = model.build_train_net(args)
            
        if args.distributed:
            optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(loss, start_prog)
    
        filelist = [args.train_data_path]
        train_loader = utils.create_dataloader(
                feed_var_list=feed,
                filelist=filelist,
                place=fluid.cpu_places(),
                batch_size=args.batch_size,
                dict_path=args.vocab_path,
                max_turn_num=args.max_turn_num,
                max_turn_len=args.max_turn_len,
                is_test=False,
                data_source=args.data_source,
                is_distributed=args.distributed)

        if args.distributed:
            train_prog = fleet.main_program

        exe.run(start_prog)
        if args.word_emb_init is not None:
            print("start loading word embedding init ...")
            if six.PY2:
                word_emb = np.array(
                        pickle.load(open(
                            args.word_emb_init, 'rb'))).astype('float32')
            else:
                word_emb = np.array(
                        pickle.load(open(
                            args.word_emb_init, 'rb'), 
                            encoding="bytes")).astype('float32')
            dam.set_word_embedding(word_emb, place)
            print("finish init word embedding  ...")
           
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog):
            _, feed, logits = model.build_test_net(args)
        test_prog = test_prog.clone(for_test=True)
        filelist = [args.valid_data_path]
        valid_loader = utils.create_dataloader(
                feed_var_list=feed,
                filelist=filelist,
                place=fluid.cpu_places(),
                batch_size=args.batch_size,
                dict_path=args.vocab_path,
                max_turn_num=args.max_turn_num,
                max_turn_len=args.max_turn_len,
                is_test=True,
                data_source=args.data_source,
                is_distributed=args.distributed)

        train_fetch = [loss]
        valid_fetch = [logits] + ["label"]

        train(args, train_prog, test_prog, exe,
                feed, train_fetch, valid_fetch,
                train_loader, valid_loader)
    
    if args.do_test:
        assert args.distributed == False 

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    
        if not os.path.exists(args.model_path):
            raise ValueError("Invalid model init path %s" % args.model_path)
        
        place = create_place(False)
        exe = fluid.Executor(place)

        test_prog, start_prog = fluid.Program(), fluid.Program()
        with fluid.program_guard(test_prog, start_prog):
            _, feed, logits = model.build_test_net(args)
            fluid.io.load_persistables(
                    executor=exe,
                    dirname=args.model_path,
                    main_program=fluid.default_main_program())

        filelist = [args.test_data_path]
        test_loader = utils.create_dataloader(
                feed_var_list=feed,
                filelist=filelist,
                place=fluid.cpu_places(),
                batch_size=args.batch_size,
                dict_path=args.vocab_path,
                max_turn_num=args.max_turn_num,
                max_turn_len=args.max_turn_len,
                is_test=True,
                data_source=args.data_source,
                is_distributed=False)

        test_fetch = [logits] + ["label"]
        test(args, test_prog, exe, feed, test_fetch, test_loader)
        

def init_dist_env():
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)

def create_place(is_distributed):
    place_idx = int(os.environ['FLAGS_selected_gpus']) if is_distributed else 0
    return fluid.CUDAPlace(place_idx)

def train(args, train_prog, test_prog, exe, feed, 
        train_fetch, valid_fetch, train_loader, valid_loader):
    print_step = 10
    save_step = 200
    step = 0
    for epoch in range(args.num_scan_data):
        start = time.time()
        for idx, sample in enumerate(train_loader()):
            ret = exe.run(
                    program=train_prog,
                    feed=sample,
                    fetch_list=train_fetch)
            if step % print_step == 0:
                print('[TRAIN] epoch=%d step=%d loss=%f' % (epoch, step, ret[0][0]))
            if step != 0 and step % save_step == 0:
                save_path = os.path.join(args.save_path, 
                        "model.epoch_{}.step_{}".format(epoch, step))
                if args.distributed:
                    if fleet.worker_index() == 0:
                        fleet.save_persistables(executor=exe, dirname=save_path)
                        print("model saved in {}".format(save_path))
                    filename = os.path.join(args.save_path,
                            "score.epoch_{}.step_{}.worker_{}".format(
                                epoch, step, fleet.worker_index()))
                else:
                    fluid.io.save_persistables(exe, save_path, train_prog)
                    filename = os.path.join(args.save_path,
                            "score.epoch_{}.step_{}".format(epoch, step))
                score_file = open(filename, 'w')
                for idx, sample in enumerate(valid_loader()):
                    ret = exe.run(
                            program=test_prog,
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
                else:
                    result = eva.evaluate_ubuntu(filename)

                print("[VALID] local result: ")
                for metric in result:
                    value, length = result[metric]
                    print("[VALID]   {}: {}".format(metric, 1.0 * value / length))

                if args.distributed:
                    if args.ext_eval:
                        dist_result = utils.dist_eval_douban(exe, result)
                    else:
                        dist_result = utils.dist_eval_ubuntu(exe, result)
                    print("[VALID] global result: ")
                    for metric in dist_result:
                        print("[VALID]   {}: {}".format(metric, dist_result[metric]))
                
            step += 1
        end = time.time()
        print("train epoch {} time: {} s".format(epoch, end - start))

def test(args, test_prog, exe, feed, fetch, loader):
    start = time.time()
    filename = os.path.join(args.save_path, "score.test")
    score_file = open(filename, 'w')
    for idx, sample in enumerate(loader()):
        ret = exe.run(
                program=test_prog,
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

    print("[TEST] local result: ")
    for metric in result:
        value, length = result[metric]
        print("[TEST]   {}: {}".format(metric, 1.0 * value / length))
    end = time.time()
    print("test time: {} s".format(end - start))
    
if __name__ == '__main__':
    main()
