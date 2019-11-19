import time
import paddle
import paddle.fluid as fluid
from network import word2vec_net
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
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
        for epoch in range(num_epochs):
            dataset.set_filelist(file_list)
            start_time = time.time()
            exe.train_from_dataset(program=fleet.main_program, dataset=dataset,
                                   fetch_list=[loss], fetch_info=['loss'],
                                   print_period=100, debug=False)
            end_time = time.time()
            if role.is_first_worker() == 0:
                model_path = str(model_path) + '/trainer_' + str(role.worker_index()) + '_epoch_' + str(epoch)
                fleet.save_persistables(executor=exe, dirname=model_path)
        logger.info("Train Success!")
        fleet.stop_worker()

if __name__ == '__main__':
    strategy = DistributeTranspilerConfig()
    strategy.sync_mode = False
    strategy.runtime_split_send_recv = True
    if is_geo_sgd:
       strategy.geo_sgd_mode = True
       strategy.geo_sgd_need_push_nums = 400
    train(strategy)
