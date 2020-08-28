import time
import paddle
import paddle.fluid as fluid
from network import word2vec_net
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

def train():
    loss, inputs = word2vec_net(dict_size, embedding_size, neg_num)
    var_dict = {'loss': loss}
    
    optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True))
    optimizer.minimize(loss)
               
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    
    dataset = get_dataset_reader(inputs)
    file_list = [str(train_files_path) + "/%s" % x for x in os.listdir(train_files_path)]
    for epoch in range(num_epochs):
        dataset.set_filelist(file_list)
        start_time = time.time() 

        class fetch_vars(fluid.executor.FetchHandler):
            def handler(self, res_dict):
                loss_value = res_dict['loss']
                logger.info(
                    "epoch -> {}, loss -> {}, at: {}".format(epoch, loss_value, time.ctime()))

        exe.train_from_dataset(program=fluid.default_main_program(), dataset=dataset,
                               fetch_handler=fetch_vars(var_dict=var_dict))
        end_time = time.time()
        model_path = str(model_path) + '/trainer_' + str(role.worker_index()) + '_epoch_' + str(epoch)
        fluid.io.save_persistables(executor=exe, dirname=model_path)
    logger.info("Train Success!")

if __name__ == '__main__':
    train()
