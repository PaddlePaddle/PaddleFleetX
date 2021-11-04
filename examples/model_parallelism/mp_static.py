import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import os
import paddle.nn as nn

paddle.enable_static()

vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4

def set_random_seed(seed, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed + rank_id)


class SimpleNet(nn.Layer):
    def __init__(self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2):

        super(SimpleNet, self).__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc1)
            ),
            bias_attr=None
        )

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc2)
            ),
            bias_attr=None
        )
        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.1)
            )
        )

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x.mean()


class SimpleMPNet(nn.Layer):
    def __init__(self, vocab_size, hidden_size, inner_size, output_size, np_fc1,
                 np_fc2, mp_id):
        super(SimpleMPNet, self).__init__()
        if mp_id == 0:
            init_fc1_data = np_fc1[:, :(inner_size // 2)]
            init_fc2_data = np_fc2[:(inner_size // 2), :]
        else:
            init_fc1_data = np_fc1[:, (inner_size // 2):]
            init_fc2_data = np_fc2[(inner_size // 2):, :]
        self.weight_attr1 = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Assign(init_fc1_data)
        )
        self.weight_attr2 = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Assign(init_fc2_data)
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.1)
            )
        )

        self.embedding_weight = paddle.nn.initializer.Constant(value=0.5)

    def forward(self, x):
        x = paddle.distributed.split(
            x, size=(vocab_size, hidden_size), operation="embedding", axis=0, num_partitions=2,
            gather_out=True, weight_attr=self.embedding_weight
        )
        x = paddle.distributed.split(
            x, size=(hidden_size, inner_size), operation="linear", axis=1, num_partitions=2,
            gather_out=False, weight_attr=self.weight_attr1
        )
        x = paddle.distributed.split(
            x, size=(inner_size, hidden_size), operation="linear", axis=0, num_partitions=2,
            gather_out=True, weight_attr=self.weight_attr2
        )
        x = self.linear3(x)
        return x.mean()


def gen_data():
    np.random.seed(2021)
    while True:
        data = [np.random.randint(0, vocab_size, [seq_length])]
        yield data

if __name__ == "__main__":
    device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
    set_random_seed(1024, device_id)
    np_fc1 = np.random.random_sample((hidden_size, inner_size))
    np_fc2 = np.random.random_sample((inner_size, hidden_size))    
    train_mp_col_program = fluid.Program()
    mp_startup_program = fluid.Program()
    strategy = fleet.DistributedStrategy()
    strategy.tensor_parallel = True
    strategy.tensor_parallel_configs = {'tensor_parallel_degree': 2}
    fleet.init(is_collective=True)
    with fluid.program_guard(main_program=train_mp_col_program, startup_program=mp_startup_program):
        data_in = fluid.data(
            name="data_in", shape=[batch_size, seq_length], dtype="int32"
        )
        train_reader = paddle.batch(gen_data, batch_size=batch_size)
        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[data_in],
            capacity=64,
            use_double_buffer=False,
            iterable=False
        )
        rank = fleet.worker_index()
        model_mp = SimpleMPNet(vocab_size, hidden_size, inner_size, output_size,
                          np_fc1, np_fc2, mp_id=rank)
        model_single = SimpleNet(vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2)        
        avg_cost_mp = model_mp(data_in)
        avg_cost_single = model_single(data_in)
        mp_opt = fluid.optimizer.SGD(0.1)
        dist_opt = fleet.distributed_optimizer(mp_opt, strategy=strategy)
        dist_opt.minimize(avg_cost_mp)
        single_opt = fluid.optimizer.SGD(0.1)
        single_opt.minimize(avg_cost_single)

    place = paddle.CUDAPlace(device_id)
    exe = paddle.static.Executor(place)
    exe.run(mp_startup_program)
    data_loader.set_sample_list_generator(train_reader, place)
    data_loader.start()
    fetch_lists = []
    fetch_lists.extend([avg_cost_mp, avg_cost_single])
    for i in range(5):
        vars = exe.run(train_mp_col_program, fetch_list=fetch_lists)
        print("mp_loss: ", vars[0], "single_loss: ", vars[1])
    data_loader.reset()
    