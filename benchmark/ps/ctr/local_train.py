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
    dataset.set_batch_size(args.batch_size)
    dataset.set_thread(args.thread_num)

    start_time = time.time()
    for i in range(args.num_passes):
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset,
                               debug=True)
        model_dir = args.model_output_dir + '/epoch' + str(i + 1) + ".model"
        fluid.io.save_inference_model(model_dir, [dense_input.name] + [x.name for x in sparse_input_ids] + [label.name], [loss, auc_var], exe)
        sys.stderr.write("epoch%d finished" % (i + 1))
    end_time = time.time()
    print("total training time: %f" % (end_time - start_time))

if __name__ == '__main__':
    train()
