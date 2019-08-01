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
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
import py_reader_generator as py_reader


class FleetRunnerBase(object):
    """
    Distribute training base class:
        This class abstracts the training process into several major steps:
        1. input_data
        2. net
        3. run_pserver
        4. run_trainer
        5. run_infer
        6. py_reader
        7. dataset_reader
        8. runtime_main
        ...
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
        raise NotImplementedError("net should be implemented by child classes.")

    def run_pserver(self, params):
        """
        Function run_pserver: Operation method of parameter server
        Args
            :params the hyper parameters of network
        Returns:
            None
        """
        role = role_maker.UserDefinedRoleMaker(
            current_id=params.current_id,
            role=role_maker.Role.SERVER,
            worker_num=params.trainers,
            server_endpoints=params.pserver_endpoints)

        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = params.sync_mode
        fleet.init(role)

        reader = None
        inputs = self.input_data(params)
        feeds = []
        if params.is_pyreader_train:
            reader = self.py_reader(params)
            inputs = fluid.layers.read_file(reader)
        elif not params.is_dataset_train:
            raise ValueError(
                "Program must has Date feed method: is_pyreader_train / is_dataset_train"
            )

        # For the model: ctr-dnn, we use loss,auc,batch_auc to measure the performance of network
        # Replace it with your network evaluation index,
        # Oops, function: self.net don't forget return the input_data, we will use it in function: self.run_infer
        loss, auc_var, batch_auc_var, _ = self.net(inputs, params)

        # define the optimizer for your model
        optimizer = fluid.optimizer.Adam(params.learning_rate)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(loss)

        fleet.init_server()
        print("PServer init success!")
        fleet.run_server()

    def run_trainer(self, params):
        """
        Function run_trainer: Operation method of training node
        Args:
            :params params: the hyper parameters of network
        Returns
            :train_result: the dict of training log
        """
        print("run trainer")

        role = role_maker.UserDefinedRoleMaker(
            current_id=params.current_id,
            role=role_maker.Role.WORKER,
            worker_num=params.trainers,
            server_endpoints=params.pserver_endpoints)

        strategy = DistributeTranspilerConfig()
        if params.is_dataset_train:
            params.sync_mode=False
            params.async_mode=True
        strategy.sync_mode = params.sync_mode
        fleet.init(role)

        reader = None
        feeds = []
        inputs = self.input_data(params)
        if params.is_pyreader_train:
            reader = self.py_reader(params)
            inputs = fluid.layers.read_file(reader)
        elif not params.is_dataset_train:
            raise ValueError(
                "Program must has Date feed method: is_pyreader_train / is_dataset_train"
            )

        # For the model: ctr-dnn, we use loss,auc,batch_auc to measure the performance of network
        # Replace it with your network evaluation index,
        # Oops, function: self.net don't forget return the input_data, we will use it in function: self.run_infer
        loss, auc_var, batch_auc_var, _ = self.net(inputs, params)

        # define the optimizer for your model
        optimizer = fluid.optimizer.Adam(params.learning_rate)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(loss)
        print("Program construction complete")

        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init_worker()
        # No need to exe.run(fluid.default_main_program())
        exe.run(fleet.startup_program)
        CPU_NUM = int(params.cpu_num)

        train_result = {}

        # Notice: Both dataset and py_reader method don't using feed={dict} to input data
        # Paddle Fluid enter data by variable name
        # When we do the definition of the reader, the program has established the workflow
        if params.is_dataset_train:
            print("run dataset train")
            dataset = self.dataset_reader(inputs, params)
            file_list = [
                str(params.train_files_path) + "/%s" % x
                for x in os.listdir(params.train_files_path)
            ]
            if params.is_local_cluster:
                file_list = fleet.split_files(file_list)
            print("file list: {}".format(file_list))

            for epoch in range(params.epochs):
                dataset.set_filelist(file_list)
                start_time = time.clock()

                # Notice: function train_from_dataset does not return fetch value
                exe.train_from_dataset(
                    program=fleet.main_program,
                    dataset=dataset,
                    fetch_list=[auc_var],
                    fetch_info=['auc'],
                    print_period=10,
                    debug=False)
                end_time = time.clock()
                self.record_time(epoch, train_result, end_time - start_time)

        elif params.is_pyreader_train:
            train_generator = py_reader.CriteoDataset(params.sparse_feature_dim)
            file_list = [
                str(params.train_files_path) + "/%s" % x
                for x in os.listdir(params.train_files_path)
            ]
            print("file list: {}".format(file_list))

            train_reader = paddle.batch(
                paddle.reader.shuffle(
                    train_generator.train(file_list, params.trainers,
                                          params.current_id),
                    buf_size=params.batch_size * 100),
                batch_size=params.batch_size)
            reader.decorate_paddle_reader(train_reader)

            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = int(params.cpu_num)
            build_strategy = fluid.BuildStrategy()
            build_strategy.async_mode = params.async_mode
            if CPU_NUM > 1:
                build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

            compiled_prog = fluid.compiler.CompiledProgram(
                fleet.main_program).with_data_parallel(
                    loss_name=loss.name,
                    build_strategy=build_strategy,
                    exec_strategy=exec_strategy)

            for epoch in range(params.epochs):
                # Notice: py_reader should use try & catch EOFException method to enter the dataset
                # reader.start() must declare in advance
                reader.start()
                start_time = time.clock()
                batch_id = 0
                try:
                    while True:
                        loss_val, auc_val, batch_auc_val = exe.run(
                            program=compiled_prog,
                            fetch_list=[
                                loss.name, auc_var.name, batch_auc_var.name
                            ])
                        loss_val = np.mean(loss_val)
                        auc_val = np.mean(auc_val)
                        batch_auc_val = np.mean(batch_auc_val)
                        if batch_id % 10 == 0 and batch_id != 0:
                            print(
                                "TRAIN --> pass: {} batch: {} loss: {} auc: {}, batch_auc: {}"
                                .format(epoch, batch_id, loss_val / params.
                                        batch_size, auc_val, batch_auc_val))
                        batch_id += 1
                except fluid.core.EOFException:
                    reader.reset()

                end_time = time.clock()
                train_result = self.record_time(epoch, train_result,
                                                end_time - start_time)

        train_method = ''
        if params.is_pyreader_train:
            train_method = '_pyreader_train'
        else:
            train_method = '_dataset_train'
        log_path = str(params.log_path + '/' + str(params.current_id) +
                       train_method + '.log')
        with open(log_path, 'w+') as f:
            f.write(str(train_result))

        if params.is_first_trainer:
            model_path = str(params.model_path + '/final' + train_method)
            fleet.save_persistables(executor=exe, dirname=model_path)

        print("Train Success!")
        fleet.stop_worker()
        return train_result

    def run_infer(self, params):
        """
        Function run_infer: Operation method of training node
        Args:
            :params params: the hyper parameters of network
        Returns
            :infer_result, type:dict, record the evalution parameter and program resource usage situation
        """
        place = fluid.CPUPlace()
        inference_scope = fluid.Scope()
        dataset = py_reader.CriteoDataset(params.sparse_feature_dim)
        file_list = [
            str(params.test_files_path) + "/%s" % x
            for x in os.listdir(params.test_files_path)
        ]
        test_reader = paddle.batch(
            dataset.test(file_list), batch_size=params.batch_size)
        startup_program = fluid.framework.Program()
        test_program = fluid.framework.Program()

        def set_zero(var_name):
            param = inference_scope.var(var_name).get_tensor()
            param_array = np.zeros(param._get_dims()).astype("int64")
            param.set(param_array, place)

        with fluid.framework.program_guard(test_program, startup_program):
            inputs = self.input_data(params)
            loss, auc_var, batch_auc_var, data_list = self.net(inputs, params)

            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(feed_list=data_list, place=place)

            train_method = ''
            if params.is_pyreader_train:
                train_method = '_pyreader_train/'
            else:
                train_method = '_dataset_train/'
            model_path = params.model_path + '/final' + train_method
            fluid.io.load_persistables(
                executor=exe,
                dirname=model_path,
                main_program=fluid.default_main_program())

            auc_states_names = ['_generated_var_2', '_generated_var_3']
            for name in auc_states_names:
                set_zero(name)

            run_index = 0
            L = []
            A = []
            for batch_id, data in enumerate(test_reader()):
                loss_val, auc_val = exe.run(test_program,
                                            feed=feeder.feed(data),
                                            fetch_list=[loss, auc_var])
                run_index += 1
                L.append(loss_val / params.batch_size)
                A.append(auc_val)
                if batch_id % 1000 == 0:
                    print("TEST --> batch: {} loss: {} auc: {}".format(
                        batch_id, loss_val / params.batch_size, auc_val))

            infer_loss = np.mean(L)
            infer_auc = np.mean(A)
            infer_result = {}
            infer_result['loss'] = infer_loss
            infer_result['auc'] = infer_auc
            log_path = params.log_path + '/infer_result.log'
            with open(log_path, 'w+') as f:
                f.write(str(infer_result))
            print("Inference complete")
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

    def runtime_main(self, params):
        """
        Function runtime_main: the entry point for program running
        Args:
            :params params: the hyper parameters of network
        """

        # Step1: get the environment variable, mainly related to network communication parameters
        params.role = os.getenv("TRAINING_ROLE")
        print("Training role: {}".format(params.role))

        params.current_id = int(os.getenv("PADDLE_TRAINER_ID"))
        print("Current Id: {}".format(params.current_id))

        params.trainers = int(os.getenv("PADDLE_TRAINERS_NUM"))
        print("Trainer num: {}".format(params.trainers))

        params.pserver_ports = os.getenv("PADDLE_PORT")
        print("Pserver ports: {}".format(params.pserver_ports))

        params.pserver_ip = os.getenv("PADDLE_PSERVERS")
        print("Pserver IP: {}".format(params.pserver_ip))

        params.current_endpoint = os.getenv(
            "POD_IP", "localhost") + ":" + params.pserver_ports

        params.cpu_num = os.getenv("CPU_NUM")
        print("output path: {}".format(params.model_path))

        if params.is_local_cluster:
            for port in params.pserver_ports.split(","):
                params.pserver_endpoints.append(':'.join(
                    [params.pserver_ip, port]))
        else:
            for ip in params.pserver_ip.split(","):
                params.pserver_endpoints.append(':'.join(
                    [ip, params.pserver_ports]))

        params.endpoints = ",".join(params.pserver_endpoints)
        #params.pserver_endpoints = params.endpoints.split(",")

        if params.role == "TRAINER" and params.current_id == 0:
            params.is_first_trainer = True

        # Step2: According to the parameters-> TRAINING_ROLE, decide which method to run
        if params.role == "PSERVER":
            self.run_pserver(params)
        elif params.role == "TRAINER":
            self.run_trainer(params)
        else:
            raise ValueError(
                "Please choice training role for current node : PSERVER / TRAINER"
            )

        # Step3: If the role is first trainer, after training, perform verification on the test data
        if params.is_first_trainer:
            infer_result = self.run_infer(params)
            infer_result_path = "./" + params.log_path + "/infer_result.txt"
            with open(infer_result_path, 'w') as f:
                f.writelines(str(infer_result))
