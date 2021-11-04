import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet


def set_random_seed(seed, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed + rank_id)


vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4


class SimpleNet(fluid.dygraph.Layer):
    def __init__(self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2):

        super(SimpleNet, self).__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc1)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            )
        )
        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc2)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            )
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
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
        return x

class SimpleMPNet(fluid.dygraph.Layer):
    def __init__(self, vocab_size, hidden_size, inner_size, output_size, np_fc1,
                 np_fc2, mp_id):
        super(SimpleMPNet, self).__init__()
        if mp_id == 0:
            init_fc1_data = np_fc1[:, :(inner_size // 2)]
            init_fc2_data = np_fc2[:(inner_size // 2), :]
        else:
            init_fc1_data = np_fc1[:, (inner_size // 2):]
            init_fc2_data = np_fc2[(inner_size // 2):, :]

        self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(init_fc1_data)
            ),
            gather_output=False,
            has_bias=True
        )        

        self.linear2 = fleet.meta_parallel.RowParallelLinear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(init_fc2_data)
            ),
            input_is_parallel=True,
            has_bias=True
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            )
        )

        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5)
        )

    def forward(self, x):        
        x = self.embedding(x)
        x = self.linear1(x) 
        x = self.linear2(x) 
        x = self.linear3(x) 
        return x

def train_batch(batch, model, optimizer):
    output = model(batch)
    loss = output.mean()
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    return loss

if __name__ == "__main__":
    paddle.distributed.init_parallel_env()
    strategy = fleet.DistributedStrategy()
    model_parallel_size = 2
    data_parallel_size = 1
    strategy.hybrid_configs = {
        "dp_degree": data_parallel_size,
        "mp_degree": model_parallel_size,
        "pp_degree": 1
    }
    fleet.init(is_collective=True, strategy=strategy)

    hcg = fleet.get_hybrid_communicate_group()
    mp_id = hcg.get_model_parallel_rank()
    rank_id = dist.get_rank()
    set_random_seed(1024, rank_id)
    np_fc1 = np.random.random_sample((hidden_size, inner_size))
    np_fc2 = np.random.random_sample((inner_size, hidden_size))
     
    model_b = SimpleNet(vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2)
    optimizer_b = paddle.optimizer.SGD(learning_rate=0.001, parameters=model_b.parameters())

    model_a = SimpleMPNet(vocab_size, hidden_size, inner_size, output_size,
                          np_fc1, np_fc2, mp_id)
    optimizer_a = paddle.optimizer.SGD(learning_rate=0.001, parameters=model_a.parameters())
    model_a = fleet.distributed_model(model_a)
    optimizer_a = fleet.distributed_optimizer(optimizer_a)


    for _ in range(5):
        np_data = np.random.randint(0, vocab_size, (batch_size, seq_length, ))
        batch = paddle.to_tensor(np_data)
        loss_a = train_batch(batch, model_a, optimizer_a)
        loss_b = train_batch(batch, model_b, optimizer_b)

        print("mp_loss: ", loss_a.numpy()[0], " single_loss: ", loss_b.numpy()[0])

    