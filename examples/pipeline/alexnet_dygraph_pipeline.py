import numpy as np
import os
import paddle
from paddle.distributed import fleet
from paddle.fluid.dygraph.container import Sequential
import paddle.nn as nn
from paddle.fluid.dygraph.layers import Layer
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
import paddle.nn.functional as F
import paddle.distributed as dist
import random


class ReshapeHelp(Layer):
    def __init__(self, shape):
        super(ReshapeHelp, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(shape=self.shape)


class AlexNet(Layer):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = Sequential(
            nn.Conv2D(
                1, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            nn.Conv2D(
                64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            nn.Conv2D(
                192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(
                384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(
                256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2), )

        self.reshape_layer = ReshapeHelp(shape=[-1, 256])
        self.classifier = nn.Linear(256, num_classes)
        self.loss_fn = nn.loss.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.features(x)
        x = self.reshape_layer(x)
        x = self.classifier(x)
        return self.loss_fn(x, y)


class AlexNetPipeDesc(PipelineLayer):
    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        decs = [
            LayerDesc(
                nn.Conv2D, 1, 64, kernel_size=11, stride=4, padding=5),
            LayerDesc(nn.ReLU),
            LayerDesc(
                nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(
                nn.Conv2D, 64, 192, kernel_size=5, padding=2),
            F.relu,
            LayerDesc(
                nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(
                nn.Conv2D, 192, 384, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(
                nn.Conv2D, 384, 256, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(
                nn.Conv2D, 256, 256, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(
                nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(
                ReshapeHelp, shape=[-1, 256]),
            LayerDesc(nn.Linear, 256, self.num_classes),  # classifier
        ]
        super(AlexNetPipeDesc, self).__init__(
            layers=decs, loss_fn=nn.CrossEntropyLoss(), **kwargs)

def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)
    print("seed: ", seed)
    print("rank_id: ", rank_id)
    print("dp_id: ", dp_id)

if __name__ == "__main__":
    # paddle.distributed.init_parallel_env()
    batch_size = 4
    micro_batch_size = 2
    
    strategy = fleet.DistributedStrategy()
    model_parallel_size = 1
    data_parallel_size = 1
    pipeline_parallel_size = 2
    strategy.hybrid_configs = {
        "dp_degree": data_parallel_size,
        "mp_degree": model_parallel_size,
        "pp_degree": pipeline_parallel_size
    }
    strategy.pipeline_configs = {
        "accumulate_steps": batch_size // micro_batch_size,
        "micro_batch_size": micro_batch_size
    }

    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    world_size = hcg.get_model_parallel_world_size()
    dp_id = hcg.get_data_parallel_rank()
    pp_id = hcg.get_stage_id()
    rank_id = dist.get_rank()
    set_random_seed(1024, dp_id, rank_id)

    model_a = AlexNet(10)
    scheduler_a = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=[2], values=[0.001, 0.002], verbose=False
    )
    optimizer_a = paddle.optimizer.SGD(learning_rate=scheduler_a,
                                       parameters=model_a.parameters())
    param_len = len(model_a.parameters())
    parameters = []
    for param in model_a.parameters():
        parameters.append(param.numpy())
    
    
    model_b = AlexNetPipeDesc(num_stages=pipeline_parallel_size, topology=hcg._topo)
    scheduler_b = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=[2], values=[0.001, 0.002], verbose=False
    )
    optimizer_b = paddle.optimizer.SGD(learning_rate=scheduler_b, 
                                        parameters=model_b.parameters())
    model_b = fleet.distributed_model(model_b)
    optimizer_b = fleet.distributed_optimizer(optimizer_b)

    for idx, param in enumerate(model_b.parameters()):
        param.set_value(parameters[idx + pp_id * (param_len // 2)])

    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=batch_size, drop_last=True
    )     

    for step_id, data in enumerate(train_reader()):
        x_data = np.array([x[0] for x in data]).astype("float32").reshape(
            batch_size, 1, 28, 28
        )
        y_data = np.array([x[1] for x in data]).astype("int64").reshape(
            batch_size, 1
        )
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        img.stop_gradient = True
        label.stop_gradient = True

        if step_id >= 5:
            break

        loss_a = model_a(img, label)
        loss_a.backward()                     
        optimizer_a.step()
        optimizer_a.clear_grad()
        scheduler_a.step()
        
        loss_b = model_b.train_batch([img, label], optimizer_b, scheduler_b)

        print("single_loss: ", loss_a.numpy(), "pp_loss: ", loss_b.numpy())



    
