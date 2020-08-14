from __future__ import print_function

from args import parse_args
import os
import paddle.fluid as fluid
import sys
import time
from network_conf import ctr_dnn_model_dataset

dense_feature_dim = 13

def train():
    args = parse_args()
    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)
    
    dense_input = fluid.layers.data(
        name="dense_input", shape=[dense_feature_dim], dtype='float32')
    sparse_input_ids = [
        fluid.layers.data(name="C" + str(i), shape=[1], lod_level=1, dtype="int64")
        for i in range(1, 27)]
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    loss, auc_var, batch_auc_var = ctr_dnn_model_dataset(dense_input, sparse_input_ids, label,
                                                         args.embedding_size, args.sparse_feature_dim)

    optimizer = fluid.optimizer.Adagrad(learning_rate=1e-2)
    optimizer.minimize(loss)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var([dense_input] + sparse_input_ids + [label])
    pipe_command = "python criteo_reader.py %d" % args.sparse_feature_dim
    dataset.set_pipe_command(pipe_command)
    whole_filelist = ["raw_data/part-%d" % x for x in range(len(os.listdir("raw_data")))]
    dataset.set_filelist(whole_filelist)
    batches = [32, 64, 128, 256, 512, 1024]
    threads = [11, 22]

    time_summary = {}
    for bs in batches:
        for thr in threads:
            start_time = time.time()
            dataset.set_batch_size(bs)
            dataset.set_thread(thr)
            exe.train_from_dataset(program=fluid.default_main_program(),
                                   dataset=dataset,
                                   debug=True)
            end_time = time.time()
            time_summary[(bs, thr)] = end_time - start_time

    total_inst = 44000000.0
    for key in time_summary:
        time_summary[key] = total_inst / time_summary[key]

    print("batch v.s threads\tthread=%d\tthread=%d" % (threads[0], threads[1]))
    for bs in batches:
        out_str = "batch=%d" % bs
        for thr in threads:
            out_str += "\t%7.4f/s" % time_summary[(bs, thr)]
        print(out_str)
if __name__ == '__main__':
    train()
