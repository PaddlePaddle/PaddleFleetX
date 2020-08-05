import time
import paddle
import paddle.fluid as fluid
from network import word2vec_net
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler.distributed_strategy import StrategyFactory
from conf import *
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def get_dataset_reader(inputs):
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    pipe_command = "python dataset_generator.py"
    dataset.set_pipe_command(pipe_command)
    dataset.set_batch_size(batch_size)
    thread_num = cpu_num
    dataset.set_thread(thread_num)
    return dataset

def train(strategy):
    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)
   
    loss, inputs = word2vec_net(dict_size, embedding_size, neg_num)
     
    optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True))

    strategy = StrategyFactory.create_async_strategy()
    # strategy = StrategyFactory.create_sync_strategy()
    # strategy = StrategyFactory.create_half_async_strategy()
    # strategy = StrategyFactory.create_geo_strategy(400)

    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(loss)
               
    if role.is_server():
        fleet.init_server()
        fleet.run_server()
    elif role.is_worker():
        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init_worker()
        exe.run(fleet.startup_program)
        
        dataset = get_dataset_reader(inputs)
        file_list = [str(train_files_path) + "/%s" % x for x in os.listdir(train_files_path)]
        if is_local_cluster:
            file_list = fleet.split_files(file_list)

        # for compiled_program
        # compiled_prog = fluid.compiler.CompiledProgram(
        #     fleet.main_program).with_data_parallel(
        #     loss_name=avg_cost.name,
        #     build_strategy=strategy.get_build_strategy(),
        #     exec_strategy=strategy.get_executor_strategy())

        # for epoch in range(num_epochs):
        #     ....

        for epoch in range(num_epochs):
            dataset.set_filelist(file_list)
            start_time = time.time()
            exe.train_from_dataset(program=fleet.main_program, dataset=dataset,
                                   fetch_list=[loss], fetch_info=['loss'],
                                   print_period=100, debug=False)
            end_time = time.time()
            if role.is_first_worker():
                model_path = str(model_path) + '/trainer_' + str(role.worker_index()) + '_epoch_' + str(epoch)
                fleet.save_persistables(executor=exe, dirname=model_path)
        logger.info("Train Success!")
        fleet.stop_worker()

if __name__ == '__main__':
    train()
