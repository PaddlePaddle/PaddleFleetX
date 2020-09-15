import os

import paddle.fluid as fluid
import paddle.fluid.distributed.downpour as downpour
import paddle.fluid.data_feed_desc as data_feed

from google.protobuf import text_format
from nets import example_net

class Model(object):

    def __init__(self):
        self.label = fluid.layers.data(name="click", shape=[-1, 1], dtype="int64", lod_level=0, append_batch_size=False)
        self.user = []
        self.item = []
        for i_slot in [str(i) for i in range(0, 50)]:
            self.item.append(fluid.layers.data(name=i_slot, shape=[1], dtype="int64", lod_level=1))
        for i_slot in [str(i) for i in range(50, 100)]:
            self.user.append(fluid.layers.data(name=i_slot, shape=[1], dtype="int64", lod_level=1))

        self.avg_cost, prediction = example_net(self.user, self.item, self.label)

        sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=0.002)

        self.startup_program = fluid.default_startup_program()
        self.program_desc = fluid.default_main_program()
        

if __name__ == "__main__":

    model = Model()

    dp = downpour.DownpourSGD(learning_rate=0.1, window=1)
    server_desc, skipped_ops = dp.minimize(model.avg_cost)

    server_desc_str = text_format.MessageToString(server_desc)

    async_exe = fluid.AsyncExecutor()
    instance = async_exe.config_distributed_nodes()

    if instance.is_server():
        async_exe.init_server(server_desc_str)
    elif instance.is_worker():
        async_exe.init_worker(server_desc_str, model.startup_program)
        local_data_dir = "./data/"
        # you can use this to download data from hadoop
        # async_exe.download_data("your_HADOOP_data_dir", local_data_dir, "fs_default_name", "ugi", 10)
        data_set = data_feed.DataFeedDesc(local_data_dir + "data_feed.proto")
        data_set.set_use_slots(["click"] + [str(i) for i in range(100)])
        file_list = filter(lambda x: x.find("part") != -1, [local_data_dir + i for i in os.listdir(local_data_dir)])
        async_exe.run(model.program_desc, data_set, file_list, 10, [model.label], "mpi")
        print "process:", instance._rankid, "train done"
    else:
        pass

    async_exe.stop()


