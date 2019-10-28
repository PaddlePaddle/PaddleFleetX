from __future__ import print_function

from args import parse_args
import os
import sys
import paddle.fluid as fluid
from network_conf import ctr_dnn_model_dataset
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

dense_feature_dim = 13
sparse_feature_dim = 10000001
batch_size = 100
thread_num = 10
embedding_size = 10

args = parse_args()

def main_function(is_local):
    dense_input = fluid.layers.data(
        name="dense_input", shape=[dense_feature_dim], dtype='float32')
    sparse_input_ids = [
        fluid.layers.data(name="C" + str(i), shape=[1], lod_level=1,
                          dtype="int64") for i in range(1, 27)]
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var([dense_input] + sparse_input_ids + [label])
    pipe_command = "python criteo_reader.py %d" % sparse_feature_dim
    dataset.set_pipe_command(pipe_command)
    dataset.set_batch_size(batch_size)
    dataset.set_thread(thread_num)
    whole_filelist = ["raw_data/part-%d" % x 
                      for x in range(len(os.listdir("raw_data")))]
    dataset.set_filelist(whole_filelist)
    loss, auc_var, batch_auc_var = ctr_dnn_model_dataset(
        dense_input, sparse_input_ids, label, embedding_size,
        sparse_feature_dim)

    exe = fluid.Executor(fluid.CPUPlace())

    def train_loop(epoch=20):
        for i in range(epoch):
            exe.train_from_dataset(program=fluid.default_main_program(),
                                   dataset=dataset,
                                   fetch_list=[auc_var],
                                   fetch_info=["auc"],
                                   debug=False)
        
    def local_train():
        optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
        optimizer.minimize(loss)
        exe.run(fluid.default_startup_program())
        train_loop()


    def dist_train():
        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = False
        optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(loss)
        if fleet.is_server():
            fleet.init_server()
            fleet.run_server()
        elif fleet.is_worker():
            fleet.init_worker()
            exe.run(fluid.default_startup_program())
            train_loop()

    if is_local:
        local_train()
    else:
        dist_train()

if __name__ == '__main__':
    main_function(args.is_local)
