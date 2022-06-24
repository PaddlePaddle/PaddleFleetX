import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet

vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4


class SimpleMPNet(fluid.dygraph.Layer):
   def __init__(self, vocab_size, hidden_size, inner_size, output_size):
      super(SimpleMPNet, self).__init__()
      self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            hidden_size,
            inner_size,
            gather_output=False,
            has_bias=True)

      self.linear2 = fleet.meta_parallel.RowParallelLinear(
            inner_size,
            hidden_size,
            input_is_parallel=True,
            has_bias=True)

      self.linear3 = paddle.nn.Linear(hidden_size, output_size)

      self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
                        vocab_size,
                        hidden_size)

   def forward(self, x):
      x = self.embedding(x)
      x = self.linear1(x)
      x = self.linear2(x)
      x = self.linear3(x)
      return x


def set_random_seed(seed, rank_id):
   random.seed(seed)
   np.random.seed(seed)
   paddle.seed(seed + rank_id)
strategy = fleet.DistributedStrategy()

# 设置两路张量模型并行
model_parallel_size = 2
data_parallel_size = 1
strategy.hybrid_configs = {
   "dp_degree": data_parallel_size,
   "mp_degree": model_parallel_size,
   "pp_degree": 1
}
# 注意strategy是这里传递的，动态图只能这里，静态图还可以在distributed_optimizer里传
fleet.init(is_collective=True, strategy=strategy)

hcg = fleet.get_hybrid_communicate_group()
mp_id = hcg.get_model_parallel_rank()
rank_id = dist.get_rank()
set_random_seed(1024, rank_id)

model = SimpleMPNet(vocab_size, hidden_size, inner_size, output_size)

optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
model = fleet.distributed_model(model)
optimizer = fleet.distributed_optimizer(optimizer)


for _ in range(5):
   np_data = np.random.randint(0, vocab_size, (batch_size, seq_length, ))

   output = model(paddle.to_tensor(np_data))
   loss = output.mean()
   loss.backward()
   optimizer.step()
   optimizer.clear_grad()
   print("loss", loss.numpy())