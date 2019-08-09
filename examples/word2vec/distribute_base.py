#!/usr/bin/python
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import sys
import os
import re
import commands
import logging
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
import py_reader_generator as py_reader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class FleetDistRunnerBase(object):
    """
    Distribute training base class:
        This class abstracts the training process into several major steps:
        1. input_data: input data of network, this function should be realized by user
        2. net: network definition, this function should be defined by user
        3. run_pserver: run pserver node in distribute environment
        4. run_trainer: run trainer, choose the way of training network according to requirement params
        5. run_infer: prediction based on the trained model
        6. py_reader: using py_reader method get data, this function should be realized by user
        7. dataset_reader: using dataset method get data, this function should be realized by user
        8. runtime_main: program entry, get the environment parameters, decide which function to call
    """

    def input_data(self, params):
        """
        Function input_data: Definition of input data format in the network
        Args:
            :params: the hyper parameters of network
        Returns:
            defined by users
        """
        raise NotImplementedError(
            "input_data should be implemented by child classes.")

    def net(self, inputs, params):
        """
        Function net: Definition of network structure
        Args:
            :inputs: input data, eg: dataset and labels. defined by funtion: self.input_data
            :params: the hyper parameters of network
        Returns:
            evaluation parameter, defined by users
        """
        raise NotImplementedError(
            "net should be implemented by child classes.")

    def infer_net(self, params):
        """
        Function net: Definition of infer network structure, This function is not required
                      if the prediction is same with the training logic
        Args:
            :params: the hyper parameters of network
        Returns:
            evaluation parameter, defined by users
         """
        raise NotImplementedError(
            "net should be implemented by child classes.")

    def run_pserver(self, params):
        """
        Function run_pserver: Operation method of parameter server
        Args
            :params the hyper parameters of network
        Returns:
            None
        """
        logger.info("run pserver")

        role = role_maker.UserDefinedRoleMaker(
            current_id=params.current_id,
            role=role_maker.Role.SERVER,
            worker_num=params.trainers,
            server_endpoints=params.pserver_endpoints)

        fleet.init(role)

        # define the model, build the pserver program
        inputs = self.input_data(params)
        if params.is_pyreader_train:
            reader = self.py_reader(params)
            inputs = fluid.layers.read_file(reader)
        elif not params.is_dataset_train:
            raise ValueError("Program must has Date feed method: is_pyreader_train / is_dataset_train")
        loss = self.net(inputs, params)

        # define the optimizer
        optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=params.learning_rate,
                decay_steps=params.decay_steps,
                decay_rate=params.decay_rate,
                staircase=True))
        optimizer = fleet.distributed_optimizer(optimizer, self.strategy)
        optimizer.minimize(loss)

        fleet.init_server()
        logger.info("PServer init success!")

        with open("pserver_train.proto", 'w') as f:
            f.write(str(fluid.default_main_program()))
        with open("pserver_startup.proto", 'w') as f:
            f.write(str(fluid.default_startup_program()))

        fleet.run_server()

    def run_dataset_trainer(self, params):
        """
        Function run_dataset_trainer: Operation method of training node
        Args:
            :params params: the hyper parameters of network
        Returns
            :train_result: the dict of training log
        """
        logger.info("run trainer")

        role = role_maker.UserDefinedRoleMaker(
            current_id=params.current_id,
            role=role_maker.Role.WORKER,
            worker_num=params.trainers,
            server_endpoints=params.pserver_endpoints)

        fleet.init(role)

        inputs = self.input_data(params)
        # For the model: word2vec, we use loss to measure the performance of network
        # Replace it with your network evaluation index
        loss = self.net(inputs, params)

        # define the optimizer for your model
        optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=params.learning_rate,
                decay_steps=params.decay_steps,
                decay_rate=params.decay_rate,
                staircase=True))
        optimizer = fleet.distributed_optimizer(optimizer, self.strategy)
        optimizer.minimize(loss)

        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init_worker()
        exe.run(fleet.startup_program)

        with open(str(params.current_id) + "_trainer_train.proto", 'w') as f:
            f.write(str(fleet.main_program))

        with open(str(params.current_id) + "_trainer_startup.proto", 'w') as f:
            f.write(str(fleet.startup_program))

        train_result = {}

        # Notice: Both dataset and py_reader method don't using feed={dict} to input data
        # Paddle Fluid enter data by variable name
        # When we do the definition of the reader, the program has established the workflow
        logger.info("run dataset train")

        dataset = self.dataset_reader(inputs, params)
        file_list = [str(params.train_files_path) + "/%s" % x
                     for x in os.listdir(params.train_files_path)]
        if params.is_local_cluster:
            file_list = fleet.split_files(file_list)
        logger.info("file list: {}".format(file_list))

        for epoch in range(params.epochs):
            dataset.set_filelist(file_list)
            start_time = time.clock()

            # Notice: function train_from_dataset does not return fetch value
            exe.train_from_dataset(program=fleet.main_program, dataset=dataset,
                                   fetch_list=[loss], fetch_info=['loss'],
                                   print_period=100, debug=False)
            end_time = time.clock()
            self.record_time(epoch, train_result, end_time - start_time)
            self.record_memory(epoch, train_result)
            if params.is_first_trainer and params.test:
                model_path = str(params.model_path) + '/trainer_' + str(params.current_id) + '_epoch_' + str(epoch)
                fleet.save_persistables(executor=exe, dirname=model_path)

        if params.is_first_trainer:
            train_method = '_dataset_train'
            log_path = str(params.log_path + '/' + str(params.current_id) + train_method + '.log')
            with open(log_path, 'w+') as f:
                f.write(str(train_result))
            model_path = str(params.model_path + '/final' + train_method)
            fleet.save_persistables(executor=exe, dirname=model_path)

        logger.info("Train Success!")
        fleet.stop_worker()
        return train_result

    def run_pyreader_trainer(self, params):
        """
        Function run_trainer: Operation method of training node
        Args:
            :params params: the hyper parameters of network
        Returns
            :train_result: the dict of training log
        """
        logger.info("run trainer")

        role = role_maker.UserDefinedRoleMaker(
            current_id=params.current_id,
            role=role_maker.Role.WORKER,
            worker_num=params.trainers,
            server_endpoints=params.pserver_endpoints)

        fleet.init(role)

        exe = fluid.Executor(fluid.CPUPlace())
        inputs = self.input_data(params)
        reader = self.py_reader(params)
        inputs = fluid.layers.read_file(reader)

        # For the model: word2vec, we use loss to measure the performance of network
        # Replace it with your network evaluation index,
        loss = self.net(inputs, params)

        # define the optimizer for your model
        optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=params.learning_rate,
                decay_steps=params.decay_steps,
                decay_rate=params.decay_rate,
                staircase=True))
        optimizer = fleet.distributed_optimizer(optimizer, self.strategy)
        optimizer.minimize(loss)
        logger.info("Program construction complete")

        fleet.init_worker()
        exe.run(fleet.startup_program)
        CPU_NUM = int(params.cpu_num)
        with open(str(params.current_id) + "_trainer_train.proto", 'w') as f:
            f.write(str(fleet.main_program))

        with open(str(params.current_id) + "_trainer_startup.proto", 'w') as f:
            f.write(str(fleet.startup_program))

        train_result = {}

        # Notice: Both dataset and py_reader method don't using feed={dict} to input data
        # Paddle Fluid enter data by variable name
        # When we do the definition of the reader, the program has established the workflow
        file_list = os.listdir(params.train_files_path)
        if params.is_local_cluster:
            file_list = fleet.split_files(file_list)
        logger.info("file list: {}".format(file_list))

        # init the py_reader
        word2vec_reader = py_reader.Word2VecReader(params.dict_path, params.train_files_path,
                                                   file_list, 0, 1)
        params.dict_size = word2vec_reader.dict_size

        np_power = np.power(np.array(word2vec_reader.id_frequencys), 0.75)
        id_frequencys_pow = np_power / np_power.sum()
        reader.decorate_tensor_provider(
            py_reader.convert_python_to_tensor(id_frequencys_pow, params.batch_size, word2vec_reader.train(), params))

        # configure local training strategy
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = int(params.cpu_num)
        exec_strategy.use_experimental_executor = True

        build_strategy = fluid.BuildStrategy()
        build_strategy.async_mode = params.async_mode
        if CPU_NUM > 1:
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

        compiled_prog = fluid.compiler.CompiledProgram(
            fleet.main_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy, exec_strategy=exec_strategy)

        for epoch in range(params.epochs):
            reader.start()
            start_time = time.clock()
            batch_id = 0
            # py_reader need use "try & catch Exception" method to load data continuously
            try:
                while True:
                    loss_val = exe.run(program=compiled_prog, fetch_list=[loss.name])
                    loss_val = np.mean(loss_val)

                    if batch_id % 10 == 0 and batch_id != 0:
                        logger.info(
                            "TRAIN --> pass: {} batch: {} loss: {} reader queue:{}".
                                format(epoch, batch_id, loss_val.mean(), reader.queue.size()))
                    batch_id += 1
            except fluid.core.EOFException:
                reader.reset()
            end_time = time.clock()

            train_result = self.record_time(epoch, train_result, end_time - start_time)
            train_result = self.record_memory(epoch, train_result)

            if params.is_first_trainer and params.test:
                model_path = str(params.model_path) + '/trainer_' + str(params.current_id) + '_epoch_' + str(epoch)
                fleet.save_persistables(executor=exe, dirname=model_path)

        if params.is_first_trainer:
            train_method = '_pyreader_train'
            log_path = str(params.log_path + '/' + str(params.current_id) + train_method + '.log')
            with open(log_path, 'w+') as f:
                f.write(str(train_result))
            model_path = str(params.model_path + '/final' + train_method)
            fleet.save_persistables(executor=exe, dirname=model_path)

        logger.info("Train Success!")
        fleet.stop_worker()
        return train_result

    def run_infer(self, params):
        """
        Function run_infer: Operation method of prediction
        Args:
            :params params: the hyper parameters of network
        Returns
            :infer_result, type:dict, record the evalution parameter and program resource usage situation
        """
        params.vocab_size, test_reader, id2word = py_reader.prepare_data(
            params.test_files_path, params.infer_dict_path, batch_size=params.infer_batch_size)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        infer_result = {}
        startup_program = fluid.framework.Program()
        test_program = fluid.framework.Program()

        with fluid.framework.program_guard(test_program,startup_program):
            values, pred = self.infer_net(params)
            train_method = ''
            if params.is_pyreader_train:
                train_method = '_pyreader_train/'
            else:
                train_method = '_dataset_train/'
            model_path = params.model_path + '/final' + train_method

            fluid.io.load_persistables(
                executor=exe, dirname=model_path, main_program=fluid.default_main_program())

            accum_num = 0
            accum_num_sum = 0.0
            step_id = 0
            for data in test_reader():
                step_id += 1
                b_size = len([dat[0] for dat in data])
                wa = np.array([dat[0] for dat in data]).astype("int64").reshape(b_size, 1)
                wb = np.array([dat[1] for dat in data]).astype("int64").reshape(b_size, 1)
                wc = np.array([dat[2] for dat in data]).astype("int64").reshape(b_size, 1)

                label = [dat[3] for dat in data]
                input_word = [dat[4] for dat in data]
                para = exe.run(fluid.default_main_program(),
                               feed={
                                   "analogy_a": wa, "analogy_b": wb, "analogy_c": wc,
                                   "all_label":
                                       np.arange(params.vocab_size).reshape(
                                           params.vocab_size, 1).astype("int64"),
                               },
                               fetch_list=[pred.name, values],
                               return_numpy=False)
                pre = np.array(para[0])
                val = np.array(para[1])
                for ii in range(len(label)):
                    top4 = pre[ii]
                    accum_num_sum += 1
                    for idx in top4:
                        if int(idx) in input_word[ii]:
                            continue
                        if int(idx) == int(label[ii][0]):
                            accum_num += 1
                        break
                if step_id % 1 == 0:
                    logger.info("step:%d %d " % (step_id, accum_num))
            acc = 1.0 * accum_num / accum_num_sum
            print("acc:%.3f " % acc)
            infer_result['acc'] = acc
        return infer_result

    def py_reader(self, params):
        """
        Function py_reader: define the data read method by fluid.layers.py_reader
        help: https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/layers_cn/io_cn.html#py-reader
        Args:
            :params params: the hyper parameters of network
        Returns:
            defined by user
        """
        raise NotImplementedError(
            "py_reader should be implemented by child classes.")

    def dataset_reader(self, inputs, params):
        """
        Function dataset_reader: define the data read method by fluid.dataset.DatasetFactory
        help: https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/dataset_cn.html#fluid-dataset
        Args:
           :params inputs: input data, eg: dataset and labels. defined by funtion: self.input_data
           :params params: the hyper parameters of network
        Returns:
           defined by user
        """
        raise NotImplementedError(
            "dataset_reader should be implemented by child classes.")

    def record_time(self, epoch, train_result, time):
        """
        record the operation time
        """
        train_result[epoch] = {}
        train_result[epoch]['time'] = time
        return train_result

    def record_memory(self, epoch, train_result):
        info = process_info()
        logger.info(info)
        train_result[epoch]['memory'] = info['mem']
        train_result[epoch]['cpu'] = info['cpu']
        train_result[epoch]['rss'] = info['rss']
        train_result[epoch]['vsa'] = info['vsa']
        return train_result

    def runtime_main(self, params):
        """
        Function runtime_main: the entry point for program running
        Args:
            :params params: the hyper parameters of network
        """

        # Step1: get the environment variable, mainly related to network communication parameters
        params.role = os.getenv("TRAINING_ROLE")
        logger.info("Training role: {}".format(params.role))

        params.current_id = int(os.getenv("PADDLE_TRAINER_ID"))
        logger.info("Current Id: {}".format(params.current_id))

        params.trainers = int(os.getenv("PADDLE_TRAINERS_NUM"))
        logger.info("Trainer num: {}".format(params.trainers))

        params.pserver_ports = os.getenv("PADDLE_PORT")
        logger.info("Pserver ports: {}".format(params.pserver_ports))

        params.pserver_ip = os.getenv("PADDLE_PSERVERS")
        logger.info("Pserver IP: {}".format(params.pserver_ip))

        params.current_endpoint = os.getenv("POD_IP", "localhost") + ":" + params.pserver_ports

        params.cpu_num = os.getenv("CPU_NUM")
        logger.info("output path: {}".format(params.model_path))

        # Step2: decide communication mode between PSERVER & TRAINER
        # recommended mode: pyreader + sync_mode / dataset + async_mode
        self.strategy = DistributeTranspilerConfig()
        if params.sync_mode:
            self.strategy.sync_mode = True
            self.strategy.runtime_split_send_recv = False
            params.async_mode = False
        elif params.half_async_mode:
            self.strategy.sync_mode = False
            params.async_mode = False
            self.strategy.runtime_split_send_recv = False
        elif params.async_mode:
            self.strategy.sync_mode = False
            params.async_mode = True
            self.strategy.runtime_split_send_recv = True

        # Step3: Configure communication IP and ports
        if params.is_local_cluster:
            for port in params.pserver_ports.split(","):
                logger.info("current port: %s" % port)
                params.pserver_endpoints.append(':'.join(
                    [params.pserver_ip, port]))
                logger.info("add pserver_endpoint:%s" % (params.pserver_endpoints))
        else:
            for ip in params.pserver_ip.split(","):
                params.pserver_endpoints.append(':'.join(
                    [ip, params.pserver_ports]))

        params.endpoints = ",".join(params.pserver_endpoints)
        logger.info("pserver_endpoints: {}".format(params.pserver_endpoints))

        if params.role == "TRAINER" and params.current_id == 0:
            params.is_first_trainer = True

        # Step4: According to the parameters-> TRAINING_ROLE, decide which method to run
        train_result = {}
        if params.role == "PSERVER":
            self.run_pserver(params)
        elif params.role == "TRAINER":
            if params.is_dataset_train:
                train_result = self.run_dataset_trainer(params)
            elif params.is_pyreader_train:
                train_result = self.run_pyreader_trainer(params)
        else:
            raise ValueError("Please choice training role for current node : PSERVER / TRAINER")

        # Step5: If the role is first trainer, after training, perform verification on the test data
        if params.is_first_trainer:
            infer_result = {}
            infer_result = self.run_infer(params)
            result = dict()
            result[0] = dict()
            result[0]['acc'] = infer_result['acc']
            result[1] = train_result[0]['time']
            result[7] = train_result[0]['cpu']
            if not params.is_local_cluster:
                if not os.path.isdir("./benchmark_logs/"):
                    os.makedirs("./benchmark_logs/")
                with open("./benchmark_logs/log_%d" % params.current_id, 'w') as f:
                    f.writelines(str(result))
            logger.info(result)

        print("Distribute train success!")


def process_info():
    pid = os.getpid()
    res = commands.getstatusoutput('ps aux|grep ' + str(pid))[1].split('\n')[0]

    p = re.compile(r'\s+')
    l = p.split(res)
    info = {'user': l[0],
            'pid': l[1],
            'cpu': l[2],
            'mem': l[3],
            'vsa': l[4],
            'rss': l[5], }
    return info
