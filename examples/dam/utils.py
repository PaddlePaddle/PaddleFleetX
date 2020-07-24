import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet

def create_dataloader(generator, feed, place, batch_size, is_test, is_distributed):
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


def sample_batch(sample):
    tensor = list(sample[0].values())[0]
    assert(isinstance(tensor, fluid.LoDTensor))
    return float(tensor.shape()[0])
