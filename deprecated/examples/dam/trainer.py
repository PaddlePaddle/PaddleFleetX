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

global fleet

class GPUTrainer(object):
    def __init__(self):
        pass

    def fit(self, model, train_loader, valid_loader, optimizer, epoch):
        print_step = max(1, 1000000 / train_loader.batch_size // (4 * 100))
        save_step = max(1, 1000000 / train_loader.batch_size // (4 * 10))
        step = 0

        place = int(os.environ["FLAGS_selected_gpus"])
        exe = paddle.fluid.Executor(place)
        for i in range(epoch):
            for idx, sample in enumerate(train_loader()):
                ret = exe.run(
                    program=model.main_program,
                    feed=sample, 
                    fetch_list=model.train_fetch)
            if step % print_step == 0:
                print("[TRAIN] epoch={} step={} loss={}".format(epoch, step, ret[0][0]))
                
            # do validation here
            if step != 0  and step % save_step == 0:
                save_path = os.path.join(args.save_path,
                                         "model.epoch_{}.step_{}".format(epoch, step))
                if fleet.worker_index() == 0:
                    fleet.save_persistables(executor=exe, dirname=save_path)
                    filename = os.path.join(args.save_path,
                                            "score.epoch_{}.step_{}.worker_{}".format(
                                                epoch, step, fleet.worker_index()))
                score_file = open(filename, "w")
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

                for metric in result:
                    value, length = result[metric]
                    print("[VALID]   {}: {}".format(metric, 1.0 * value / length))
                with open(os.path.join(args.save_path,
                                       "result.epoch_{}.step_{}".format(epoch, step)), "w") as f:
                    for k, v, in dist_result.items():
                        f.write("{}: {}".format(k, v) + "\n")
            step += 1
