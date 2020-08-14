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
import os
import re
import time
import numpy as np
import commands
import paddle
import logging
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
import py_reader_generator as py_reader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class FleetRunnerBase(object):
    """
    Distribute training base class:
        This class abstracts the training process into several major steps:
        1. input_data
        2. net
        3. run_pserver
        4. run_dataset_trainer
        5. run_pyreader_trainer
        6. run_infer
        7. py_reader
        8. dataset_reader
        9. runtime_main
        ...
    """

    def input_data(self, params):
        """
        Function input_data: Definition of input data format in the network
        :param params: the hyper parameters of network
        :return: defined by users
        """
        raise NotImplementedError(
            "input_data should be implemented by child classes.")

    def net(self, inputs, params):
        """
        Function net: Definition of network structure
        :param inputs: input data, eg: dataset and labels. defined by funtion: self.input_data
        :param params: the hyper parameters of network
        :return: evaluation parameter, defined by users
        """
        raise NotImplementedError(
            "net should be implemented by child classes.")

    def run_pserver(self, params):
        """
        Function run_pserver: Operation method of parameter server
        :param params: the hyper parameters of network
        """
        # step1: define the role of node, configure communication parameter
        role = role_maker.UserDefinedRoleMaker(
            current_id=params.current_id,
            role=role_maker.Role.SERVER,
            worker_num=params.trainers,
            server_endpoints=params.pserver_endpoints)
        fleet.init(role)

        # step2: define the input data of network
        reader = None
        inputs = self.input_data(params)
        if params.is_pyreader_train:
            reader = self.py_reader(params)
            inputs = fluid.layers.read_file(reader)
        elif not params.is_dataset_train:
            raise ValueError("Program must has Date feed method: is_pyreader_train / is_dataset_train")

        # step3: define the network
        # For the model: simnet, we use avg_cost, acc to measure the performance of network
        # Replace it with your network evaluation index,
        avg_cost, acc, cos_q_pt = self.net(inputs, params)

        # step4: define the optimizer for your model
        # define the optimizer for your model
        optimizer = fluid.optimizer.Adam(params.learning_rate)
        optimizer = fleet.distributed_optimizer(optimizer, self.strategy)
        optimizer.minimize(avg_cost)

        fleet.init_server()
        print("PServer init success!")
        fleet.run_server()

    def run_dataset_trainer(self, params):
        """
        Function run_dataset_trainer: Operation method of training node
        :param params: the hyper parameters of network

        """
        # step1: define the role of node, configure communication parameter
        role = role_maker.UserDefinedRoleMaker(
            current_id=params.current_id,
            role=role_maker.Role.WORKER,
            worker_num=params.trainers,
            server_endpoints=params.pserver_endpoints)
        fleet.init(role)

        # step2: define the input data of network
        inputs = self.input_data(params)

        # step3: define the network, same with PSERVER
        # For the model: simnet, we use avg_cost, acc to measure the performance of network
        # Replace it with your network evaluation index,
        avg_cost, acc, cos_q_pt = self.net(inputs, params)

        # step4: define the optimizer for your model
        optimizer = fluid.optimizer.Adam(params.learning_rate)
        optimizer = fleet.distributed_optimizer(optimizer, self.strategy)
        optimizer.minimize(avg_cost)

        # step5: define Executor and run startup program
        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init_worker()
        # No need to exe.run(fluid.default_main_program())
        exe.run(fleet.startup_program)

        # step6: init dataset reader
        # Notice: Both dataset and py_reader method don't using feed={dict} to input data
        # Paddle Fluid enter data by variable name
        # When we do the definition of the reader, the program has established the workflow
        dataset = self.dataset_reader(inputs, params)
        file_list = [str(params.train_files_path) + "/%s" % x
                     for x in os.listdir(params.train_files_path)]
        if params.is_local_cluster:
            file_list = fleet.split_files(file_list)
        logger.info("file list: {}".format(file_list))
        logger.info('----------------------NO.%s trainer ready----------------' % (params.current_id))

        # step7: begin to train your model, good luck
        train_result = {}
        for epoch in range(params.epochs):
            dataset.set_filelist(file_list)
            start_time = time.clock()
            # Notice: function train_from_dataset does not return fetch value
            exe.train_from_dataset(
                program=fleet.main_program,
                dataset=dataset,
                fetch_list=[acc],
                fetch_info=['acc'],
                print_period=100,
                debug=False)
            end_time = time.clock()
            self.record_time(epoch, train_result, end_time - start_time)
            self.record_memory(epoch, train_result)
            logger.info("epoch %d finished, use time=%d\n" % ((epoch), end_time - start_time))
            if params.is_first_trainer:
                model_path = str(params.model_path) + '/trainer_' + str(params.current_id) + '_epoch_' + str(epoch)
                fleet.save_persistables(executor=exe, dirname=model_path)

        if params.is_first_trainer:
            train_method = '_dataset_train'
            model_path = str(params.model_path + '/final' + train_method)
            fleet.save_persistables(executor=exe, dirname=model_path)

        logger.info("Train Success!")
        fleet.stop_worker()
        return train_result

    def run_pyreader_trainer(self, params):
        """
        Function run_pyreader_trainer: Operation method of training node
        :param params: the hyper parameters of network

        """
        # step1: define the role of node, configure communication parameter
        role = role_maker.UserDefinedRoleMaker(
            current_id=params.current_id,
            role=role_maker.Role.WORKER,
            worker_num=params.trainers,
            server_endpoints=params.pserver_endpoints)
        fleet.init(role)

        # step2: define the input data of network
        #inputs = self.input_data(params)
        reader = self.py_reader(params)
        inputs = fluid.layers.read_file(reader)

        # step3: define the network, same with PSERVER
        # For the model: simnet, we use avg_cost, acc to measure the performance of network
        # Replace it with your network evaluation index,
        avg_cost, acc, cos_q_pt = self.net(inputs, params)

        # step4: define the optimizer for your model
        # define the optimizer for your model
        optimizer = fluid.optimizer.SGD(params.learning_rate)
        optimizer = fleet.distributed_optimizer(optimizer, self.strategy)
        optimizer.minimize(avg_cost)

        # step5: define Executor and run startup program
        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init_worker()
        # No need to exe.run(fluid.default_main_program())
        exe.run(fleet.startup_program)

        # step6: init py_reader reader
        # Notice: Both dataset and py_reader method don't using feed={dict} to input data
        # Paddle Fluid enter data by variable name
        # When we do the definition of the reader, the program has established the workflow
        file_list = [str(params.train_files_path) + "/%s" % x
                     for x in os.listdir(params.train_files_path)]
        if params.is_local_cluster:
            file_list = fleet.split_files(file_list)
        logging.info("file list: {}".format(file_list))
        train_generator = py_reader.get_batch_reader(file_list,
                                                     batch_size=params.batch_size,
                                                     sample_rate=params.sample_rate)
        reader.decorate_paddle_reader(train_generator)

        # step7: define the compiled program
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = int(params.cpu_num)
        build_strategy = fluid.BuildStrategy()
        build_strategy.async_mode = self.async_mode
        if int(params.cpu_num) > 1:
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        compiled_prog = fluid.compiler.CompiledProgram(
            fleet.main_program).with_data_parallel(
            loss_name=avg_cost.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
        logger.info('----------------------NO.%s trainer ready----------------' % (params.current_id))

        # step8: begin to train your model, good luck
        train_result = {}
        for epoch in range(params.epochs):
            # Notice: py_reader should use try & catch EOFException method to enter the dataset
            # reader.start() must declare in advance
            reader.start()
            start_time = time.clock()
            batch_id = 0
            try:
                while True:
                    step_start = time.time()
                    cost_val, acc_val = exe.run(
                        program=compiled_prog,
                        fetch_list=[
                            avg_cost.name, acc.name])
                    cost_val = np.mean(cost_val)
                    acc_val = np.mean(acc_val)
                    step_end = time.time()
                    samples = params.batch_size * params.cpu_num

                    if batch_id % 10 == 0 and batch_id != 0:
                        print(
                            "Epoch: {0}, Step: {1}, Loss: {2}, Accuracy: {3}, Samples/sec: {4} "
                            "Train total expend: {5}, py_reader.queue.size: {6}".format(
                                epoch, batch_id, cost_val.mean(), acc_val.mean(),
                                int(samples / (step_end - step_start)),
                                step_end - step_start, reader.queue.size()))
                    batch_id += 1

            except fluid.core.EOFException:
                reader.reset()

            end_time = time.clock()
            train_result = self.record_time(epoch, train_result, end_time - start_time)
            train_result = self.record_memory(epoch, train_result)
            logger.info("epoch %d finished, use time=%d\n" % ((epoch), end_time - start_time))
            if params.is_first_trainer:
                model_path = str(params.model_path) + '/trainer_' + str(params.current_id) + '_epoch_' + str(epoch)
                fleet.save_persistables(executor=exe, dirname=model_path)

        if params.is_first_trainer:
            train_method = '_pyreader_train'
            model_path = str(params.model_path + '/final' + train_method)
            fleet.save_persistables(executor=exe, dirname=model_path)

        logger.info("Train Success!")
        fleet.stop_worker()
        return train_result

    def run_infer(self, params):
        """
        Function run_infer: Operation method of training node
        :param params: the hyper parameters of network
        :return: infer_result, type:dict, record the evalution parameter and program resource usage situation
        """
        print("start infer")
        place = fluid.CPUPlace()
        file_list = [str(params.test_files_path) + "/%s" % x
                     for x in os.listdir(params.test_files_path)]
        test_reader = py_reader.get_infer_batch_reader(file_list,
                                                       batch_size=params.batch_size,
                                                       sample_rate=params.sample_rate)
        startup_program = fluid.framework.Program()
        test_program = fluid.framework.Program()

        with fluid.framework.program_guard(test_program, startup_program):
            inputs = self.input_data(params)
            avg_cost, acc, cos_q_pt, = self.net(inputs, params)

            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(feed_list=inputs, place=place)

            train_method = ''
            if params.is_pyreader_train:
                train_method = '_pyreader_train/'
            else:
                train_method = '_dataset_train/'
            model_path = params.model_path + '/final' + train_method
            fluid.io.load_persistables(
                executor=exe,
                dirname=model_path,
                main_program=fluid.default_main_program()
            )
            run_index = 0
            L = []
            A = []

            for batch_id, data in enumerate(test_reader()):
                loss_val, acc_val = exe.run(test_program,
                                            feed=feeder.feed(data),
                                            fetch_list=[avg_cost, acc])
                run_index += 1
                loss_val = np.mean(loss_val)
                acc_val = np.mean(acc_val)
                L.append(loss_val)
                A.append(acc_val)
                if batch_id % 100 == 0:
                    print("TEST --> batch: {} loss: {} auc: {}".format(
                        batch_id, loss_val.mean(), acc_val.mean()))

            infer_loss = np.mean(L)
            infer_auc = np.mean(A)
            infer_result = {}
            infer_result['loss'] = infer_loss
            infer_result['acc'] = infer_auc
            log_path = params.log_path + '/infer_result.log'
            with open(log_path, 'w+') as f:
                f.write(str(infer_result))
        return infer_result

    def py_reader(self, params):
        """
        Function py_reader: define the data read method by fluid.layers.py_reader
        help: https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/layers_cn/io_cn.html#py-reader
        :param params: the hyper parameters of network
        :return: defined by user
        """
        raise NotImplementedError(
            "dataset_reader should be implemented by child classes.")

    def dataset_reader(self, inputs, params):
        """
        Function dataset_reader: define the data read method by fluid.dataset.DatasetFactory
        help: https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/dataset_cn.html#fluid-dataset
        :param inputs: input data, eg: dataset and labels. defined by funtion: self.input_data
        :param params: the hyper parameters of network
        :return: defined by user
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
        :param params: the hyper parameters of network
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

        params.cpu_num = int(os.getenv("CPU_NUM"))
        logger.info("cpu num: {}".format(params.cpu_num))

        # Step2: decide communication mode between PSERVER & TRAINER
        # recommended mode: pyreader + sync_mode / dataset + async_mode
        self.strategy = DistributeTranspilerConfig()
        if params.sync_mode == 'sync':
            self.strategy.sync_mode = True
            self.strategy.runtime_split_send_recv = False
            self.async_mode = False
            params.batch_size = int(params.batch_size / params.trainers)
        elif params.sync_mode == 'half_async':
            self.strategy.sync_mode = False
            self.async_mode = False
            self.strategy.runtime_split_send_recv = False
        elif params.sync_mode == 'async' or params.is_dataset_train:
            self.strategy.sync_mode = False
            self.async_mode = True
            self.strategy.runtime_split_send_recv = True

        # Step3: Configure communication IP and ports
        # If we use local cluster simulate real distributed environment:
        # -- PSERVER have same IP but different port
        # In the real distributed cluster computing environment:
        # -- PSERVER have same port but different IP
        if params.is_local_cluster:
            for port in params.pserver_ports.split(","):
                params.pserver_endpoints.append(':'.join(
                    [params.pserver_ip, port]))
        else:
            for ip in params.pserver_ip.split(","):
                params.pserver_endpoints.append(':'.join(
                    [ip, params.pserver_ports]))

        params.endpoints = ",".join(params.pserver_endpoints)
        logger.info("pserver_endpoints: {}".format(params.pserver_endpoints))

        if params.role == "TRAINER" and params.current_id == 0:
            params.is_first_trainer = True

        # Step4: According to the environment parameters-> TRAINING_ROLE, decide which method to run
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
        result = dict()
        infer_result = {}
        if params.is_first_trainer:
            infer_result = self.run_infer(params)
            result[0] = dict()
            result[0]['loss'] = infer_result['loss']
            result[0]['acc'] = infer_result['acc']
            result[1] = train_result[0]['time']
        elif params.role == "TRAINER" and params.current_id != 0:
            result[1] = train_result[0]['time']
        result_path = params.log_path + '/' + str(params.current_id) + '_result.log'
        with open(result_path, 'w') as f:
            f.write(str(result))

        logger.info("Distribute train success!")


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
