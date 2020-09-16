# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os


def is_first_worker():
    PADDLE_TRAINER_ENDPOINTS = os.environ.get('PADDLE_TRAINER_ENDPOINTS')
    current_endpoint = os.environ.get('PADDLE_CURRENT_ENDPOINT')
    if (PADDLE_TRAINER_ENDPOINTS is None) or (current_endpoint is None):
        return True
    endpoints = PADDLE_TRAINER_ENDPOINTS.split(",")
    hostname, _ = current_endpoint.split(":")
    host_endpoints = [x for x in endpoints if x.split(":")[0] == hostname]
    return host_endpoints[0] == current_endpoint

def get_node_info():
    PADDLE_TRAINER_ENDPOINTS = os.environ.get('PADDLE_TRAINER_ENDPOINTS')
    endpoints = PADDLE_TRAINER_ENDPOINTS.split(",")
    hosts = []
    current_endpoint = os.environ.get('PADDLE_CURRENT_ENDPOINT')
    current_host = current_endpoint.split(":")[0]
    for endpoint in endpoints:
        hostname, _ = endpoint.split(":")
        if hostname not in hosts:
            hosts.append(hostname)
    return hosts.index(current_host), len(hosts)

