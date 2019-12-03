本例将以一个十分简单的网络fit a line来做示例，说明如何从单机训练转换为fleet分布式训练。

网络结构定义:
```
# Configure the neural network.
def net(x, y):
    ...
    return y_predict, avg_cost, auc_var, auc_batch_var
```


数据读取定义（本例中使用numpy生成的假数据）:
```
def fake_reader():
    def reader():
            ...
            yield x,y
    return reader
```


单机训练:
```
# Define train function.
def train():
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='int64')
    y_predict, avg_cost, auc, auc_batch = net(x, y)
    sgd_optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)
    train_reader = paddle.batch(fake_reader(), batch_size=24)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    def train_loop(main_program, startup_program, is_chief=False):
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe.run(startup_program)

        PASS_NUM = 10
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                avg_loss_value, auc_value = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost, auc])
                print("Pass %d, total avg cost = %f, auc = %f" % (pass_id, avg_loss_value, auc_value))
        if is_chief:
            fluid.io.save_persistables("tmp", exe)

    train_loop(fluid.default_main_program(), fluid.default_startup_program(), true)
```


启动单机训练任务(将上述代码整合成train.py)：
# 将上述代码整合成train.py
python train.py
从单机训练任务转换为Fleet分布式训练任务：
网络结构定义、数据读取定义不需要做改动， 运行代码文件为：train-3.py

需要添加fleet相关的接口及实现到训练的代码中：

```
# Define train function.
def train():
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='int64')
    y_predict, avg_cost, auc, auc_batch = net(x, y)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    role = role_maker.PaddleCloudRoleMaker()
    config = DistributeTranspilerConfig()
    config.sync_mode = True

     # 加入 fleet init 初始化环境
    fleet.init(role)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    # 加入 fleet distributed_optimizer 加入分布式策略配置及多机优化
    optimizer = fleet.distributed_optimizer(optimizer, config)
    optimizer.minimize(avg_cost)

    # 启动server
    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()

    # 启动worker
    if fleet.is_worker():
	    # 初始化worker配置
        fleet.init_worker()

        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        train_reader = paddle.batch(fake_reader(), batch_size=24)

	    def train_loop(main_program, startup_program, is_chief=False):
    	    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        	exe.run(startup_program)

	        PASS_NUM = 10
    	    for pass_id in range(PASS_NUM):
        	    for batch_id, data in enumerate(train_reader()):
            	    avg_loss_value, auc_value = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost, auc])
                	print("Pass %d, total avg cost = %f, auc = %f" % (pass_id, avg_loss_value, auc_value))
	        if is_chief:
    	        fleet.save_persistables("tmp", exe)

		train_loop(fleet.main_program, fleet.startup_program, true)
# 通知server，当前节点训练结束
fleet.stop_worker()
```


启动1X1的多机训练任务(将上述代码整合成train.sh)：
```
#!/bin/bash

export PADDLE_TRAINERS=1
export PADDLE_TRAINER_ID=0
export PADDLE_PSERVER_PORTS=36001
export PADDLE_PSERVER_IP=127.0.0.1

if [ "$1" = "ps" ]
then
    export PADDLE_TRAINING_ROLE=PSERVER

    export GLOG_v=0
    export GLOG_logtostderr=1

    echo "PADDLE WILL START PSERVER ..."
    stdbuf -oL python train.py &> pserver.0.log &
fi

if [ "$1" = "tr" ]
then
    export PADDLE_TRAINING_ROLE=TRAINER

    export GLOG_v=0
    export GLOG_logtostderr=1

    echo "PADDLE WILL START TRAINER ..."
    stdbuf -oL python train.py &> trainer.0.log &
fi
```

