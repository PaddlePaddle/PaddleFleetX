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
"""
argument.py :define all parameters used in training and infer
"""
import argparse


def params_args(args=None):
    """
    Parse command line arguments
    :param args: command line arguments or None (default)
    :return: dictionary of parameters
    """
    # parameters of model and files
    params = argparse.ArgumentParser(description='Run distribute model CE test.')
    params.add_argument("--name", type=str, default="gru",
                        help="The name of current model")
    params.add_argument("--train_files_path", type=str, default="train_data",
                        help="Data file(s) for training.")
    params.add_argument("--test_files_path", type=str, default="test_data",
                        help="Data file(s) for validation or evaluation.")
    params.add_argument("--log_path", type=str, default="result")
    params.add_argument("--model_path", type=str, default="model")

    # parameters of training
    params.add_argument("-l", "--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate for training.")

    params.add_argument("-b", "--batch_size", type=int, default=1000,
                        help="Mini batch size for training.")
    params.add_argument("-e", "--epochs", type=int, default=20,
                        help="Number of epochs for training.")

    # customized parameters
    params.add_argument('--embedding_size', type=int, default=10,
                        help="The size for embedding layer (default:10)")
    params.add_argument('--sparse_feature_dim', type=int, default=1000001,
                        help='sparse feature hashing space for index processing')
    params.add_argument('--dense_feature_dim', type=int, default=13)

    # parameters of train method
    params.add_argument("--is_pyreader_train", type=bool, default=False)
    params.add_argument("--is_dataset_train", type=bool, default=False)
    params.add_argument('--is_local', type=int, default=1,
                        help='Local train or distributed train (default: 1)')
    params.add_argument('--test',type=bool,default=False,
                        help='support model save and upload')
    params.add_argument('--is_local_cluster', type=bool, default=False)
    
    params.add_argument('-r', "--role", type=str, required=False, choices=['TRAINER', 'PSERVER'])
    params.add_argument("--endpoints", type=str, default="",
                        help='The pserver endpoints, like: 127.0.0.1:6000,127.0.0.1:6001')
    params.add_argument('--current_endpoint', type=str, default='',
                        help='The path for model to store (default: 127.0.0.1:6000)')
    params.add_argument('-i', "--current_id", type=int, default=0,
                        help="Specifies the number of the current role")
    params.add_argument("--trainers", type=int, default=1,
                        help="Specify the number of nodes participating in the training")
    params.add_argument("--is_first_trainer", type=bool, default=False)
    params.add_argument("--pserver_ip", type=str, default="127.0.0.1")
    params.add_argument("--pserver_endpoints", type=list, default=[])
    params.add_argument("--pserver_ports", type=str, default="36001")
    params.add_argument("--sync_mode", type=bool, default=False)
    params.add_argument("--half_sync_mode",type=bool,default=False)
    params.add_argument("--async_mode", type=bool, default=False)
    params.add_argument("--runtime_split_send_recv",type=bool,default=False)
    
    params.add_argument("--cpu_num", type=int, default=11)
    params.add_argument("--use_cuda", type=bool, default=False)

    params.add_argument("--test_model_dir", type=str, default="")
    params.add_argument("--ready_path", type=str, default="")
    params.add_argument("--barrier_level", type=int, default=1)
    params = params.parse_args()
    return params
