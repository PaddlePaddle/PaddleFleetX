#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import argparse


def params_args(args=None):
    """
    Parse command line arguments
    Args:
        args command line arguments or None (default)
    Returns:
        dictionary of parameters with argparse class
    """
    # parameters of model and files
    params = argparse.ArgumentParser(
        description='Run distribute model CTR DNN.')
    params.add_argument("--name",
                        type=str,
                        default="ctr-dnn",
                        help="The name of current model")
    params.add_argument("--train_files_path",
                        type=str,
                        default="train_data",
                        help="Data file(s) for training.")
    params.add_argument("--test_files_path",
                        type=str,
                        default="test_data",
                        help="Data file(s) for validation or evaluation.")
    params.add_argument("--log_path", type=str, default="result")
    params.add_argument("--model_path", type=str, default="model")

    # parameters of training
    params.add_argument("-l",
                        "--learning_rate",
                        type=float,
                        default=1e-4,
                        help="Initial learning rate for training.")
    params.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=1000,
                        help="Mini batch size for training.")
    params.add_argument("-e",
                        "--epochs",
                        type=int,
                        default=1,
                        help="Number of epochs for training.")

    # customized parameters
    params.add_argument('--embedding_size',
                        type=int,
                        default=10,
                        help="The size for embedding layer (default:10)")
    params.add_argument('--sparse_feature_dim', type=int, default=1000001)
    params.add_argument('--dense_feature_dim', type=int, default=13)

    # parameters of train method
    params.add_argument("--test",
                        type=bool,
                        default=False,
                        help="Decide whether to save the model")
    params.add_argument("--cloud",
                        type=int,
                        default=0,
                        help="Training on cloud or local")

    # parameters of distribute
    params.add_argument("--is_sparse", type=bool, default=True)
    params.add_argument("--cpu_num", type=int, default=2)

    params = params.parse_args()
    return params
