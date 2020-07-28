import paddle.fluid as fluid
#from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import os

def load_filelist(filelist_path, is_distributed):
    if not os.path.exists(filelist_path):
        raise SystemExit("{} not exists.".format(filelist_path))

    files = []
    with open(filelist_path) as fs:
        for idx, line in enumerate(fs):
            line = line.strip()
            if is_distributed:
                rank = fleet.worker_index()
                nranks = fleet.worker_num()
                if idx % nranks == rank:
                    files.append(line)
            else:
                files.append(line)
    return files

def create_dataset(feed_var_list, filelist, batch_size, 
        thread_num, dict_path, max_turn_num, max_turn_len, 
        data_source):
    dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
    dataset.set_batch_size(batch_size)
    dataset.set_filelist(filelist)
    dataset.set_use_var(feed_var_list)
    pipe_command = "python data_generator.py {} {} {} {}".format(
            dict_path, max_turn_num, max_turn_len, data_source)
    dataset.set_pipe_command(pipe_command)
    return dataset

def create_dataloader(feed_var_list, filelist, place, batch_size, thread_num,
        dict_path, max_turn_num, max_turn_len, is_test, data_source):
    dataset = create_dataset(feed_var_list, filelist,
            batch_size, thread_num, dict_path, max_turn_num,
            max_turn_len, data_source)
    loader = fluid.io.DataLoader.from_dataset(dataset, place, drop_last=(not is_test))
    return loader

def b_create_dataloader(generator, feed, place, batch_size, is_test, is_distributed):
    def _dist_wrapper(generator):
        def _wrapper():
            rank = fleet.worker_index()
            nranks = fleet.worker_num()
            for idx, sample in enumerate(generator()):
                if idx % nranks == rank:
                    yield sample
        return _wrapper

    if is_distributed:
        generator = _dist_wrapper(generator)

    drop_last = False if is_test else True
    loader = fluid.io.DataLoader.from_generator(feed_list=feed, capacity=16)
    loader.set_sample_generator(generator,
            batch_size=batch_size,
            drop_last=drop_last,
            places=[place])
    return loader

def dist_eval_acc(exe, local_value, local_weight):
    prog = fluid.Program()
    with fluid.program_guard(prog):
        value = fluid.layers.data(name='value', shape=[1], dtype='float32')
        weight = fluid.layers.data(name='weight', shape=[1], dtype='float32')
        dist_value = fluid.layers.collective._c_allreduce(value, reduce_type='sum', use_calc_stream=True)
        dist_weight = fluid.layers.collective._c_allreduce(weight, reduce_type='sum', use_calc_stream=True)
    value_sum, weight_sum = exe.run(prog, feed={'value': local_value, 'weight': local_weight}, fetch_list=[dist_value, dist_weight]) 
    return value_sum / weight_sum
