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
from paddle.distributed.fleet.utils.fs import HDFSClient
import time
import paddle.distributed.fleet as fleet
import socket
import sys
import hashlib
from .barrier_server_impl import BarrierServer
from .barrier_client_impl import BarrierClient
from .env import is_first_worker, get_node_info
import sysconfig
import multiprocessing
import yaml
import os


def net_is_used(port, ip='127.0.0.1'):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, port))
        s.shutdown(2)
        return True
    except:
        return False


def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_exists(filelist, local_path):
    with open("{}/filelist.txt".format(local_path), 'r') as fin:
        for line in fin:
            current_file = line.split(' ')[0]
            current_md5 = line.split(' ')[1].strip()
            if current_file in filelist:
                if (not os.path.exists("{}/{}".format(
                        local_path, current_file))) or get_md5("{}/{}".format(
                            local_path, current_file)) != current_md5:
                    return True
        return False


def get_file_shard(node_id, node_num, local_path):
    while not os.path.exists('{}/filelist.txt'.format(local_path)):
        time.sleep(3)
    full_list = []
    with open("{}/filelist.txt".format(local_path), 'r') as fin:
        for line in fin:
            full_list.append(line.split(' ')[0])
    return full_list[node_id::node_num]


class Downloader(object):
    def __init__(self):
        self.need_barrier = False
        if os.environ.get('PADDLE_TRAINER_ENDPOINTS') is not None:
            endpoints = os.environ.get('PADDLE_TRAINER_ENDPOINTS').split(",")
            current_endpoint = os.environ.get('PADDLE_CURRENT_ENDPOINT')
            self.server_endpoint = endpoints[0]
            self.server_port = self.server_endpoint.split(":")[1]
            self.barrier_server = BarrierServer()
            if current_endpoint == self.server_endpoint:
                while net_is_used(self.server_port):
                    time.sleep(3)
                self.barrier_server.start_server_in_background(
                    endpoint=self.server_endpoint, worker_endpoints=endpoints)
            self.need_barrier = True

    def download_from_hdfs(self,
                           fs_yaml=None,
                           local_path="./",
                           shard_num=-1,
                           shard_id=-1,
                           process_num=10):
        """
        Download from hdfs
        The configurations are configured in fs_yaml file:
        TODO: add example and yaml argument fields introduction
        """

        def multi_download(client,
                           hdfs_path,
                           local_path,
                           filelist,
                           process_num=process_num):
            def _subprocess_download(files):
                for ff in files:
                    client.download('{}/{}'.format(hdfs_path, ff),
                                    '{}/{}'.format(local_path, ff))
                    cmd = "tar -xf {}/{} -C {}".format(local_path, ff,
                                                       local_path)
                    os.system(cmd)

            dir_per_process = len(filelist) / process_num

            procs = []
            for i in range(process_num):
                process_filelist = filelist[i::process_num]
                p = multiprocessing.Process(
                    target=_subprocess_download, args=(process_filelist, ))
                procs.append(p)
                p.start()

            for proc in procs:
                proc.join()

        if is_first_worker():
            if not os.path.exists(local_path):
                os.system('mkdir {}'.format(local_path))
        _, ext = os.path.splitext(fs_yaml)
        assert ext in ['.yml', '.yaml'], "only support yaml files for now"
        with open(fs_yaml) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        if "hadoop_home" in cfg:
            self.hadoop_home = cfg["hadoop_home"]
        elif "HADOOP_HOME" in os.environ:
            self.hadoop_home = os.environ['HADOOP_HOME']
        elif os.system('which hadoop') == 0:
            path = os.popen("which hadoop").readlines()[0].rstrip()
            self.hadoop_home = os.path.dirname(os.path.dirname(path))

        if self.hadoop_home:
            print("HADOOP_HOME: " + self.hadoop_home)

            if "fs.default.name" in cfg and "hadoop.job.ugi" in cfg:
                self.hdfs_configs = {
                    "fs.default.name": cfg["fs.default.name"],
                    "hadoop.job.ugi": cfg["hadoop.job.ugi"]
                }
        java_home = ''
        if "java_home" in cfg:
            java_home = cfg['java_home']
        os.environ['JAVA_HOME'] = java_home
        if "data_path" in cfg:
            hdfs_path = cfg["data_path"]
        else:
            raise Exception("ERROR: Please figure your data path in AFS.")
        client = HDFSClient(self.hadoop_home, self.hdfs_configs)
        if is_first_worker():
            if not (client.is_exist('{}/meta.txt'.format(hdfs_path)) and
                    client.is_exist('{}/filelist.txt'.format(hdfs_path))):
                raise Exception(
                    "ERROR: Your data dir should include filelist.txt and meta.txt"
                )
            client.download('{}/filelist.txt'.format(hdfs_path),
                            '{}/filelist.txt'.format(local_path))
            client.download('{}/meta.txt'.format(hdfs_path),
                            '{}/meta.txt'.format(local_path))
            with open('{}/meta.txt'.format(local_path), 'r') as fin:
                for line in fin:
                    current_file = line.strip()
                    client.download('{}/{}'.format(hdfs_path, current_file),
                                    '{}/{}'.format(local_path, current_file))

        if shard_num > 0:
            assert (
                shard_id >= 0,
                "Please provide worker index by fleet.worker_index() if you want to download sharded data on each machine"
            )
            self.filelist = get_file_shard(shard_id, shard_num, local_path)
            need_download = check_exists(self.filelist, local_path)
            if need_download:
                multi_download(client, hdfs_path, local_path, self.filelist)
        else:
            if is_first_worker():
                self.filelist = get_file_shard(0, 1, local_path)
                need_download = check_exists(self.filelist, local_path)
                if need_download:
                    multi_download(client, hdfs_path, local_path,
                                   self.filelist)
        if self.need_barrier:
            client = BarrierClient()
            client.server_endpoint = self.server_endpoint
            client.my_endpoint = os.environ.get('PADDLE_CURRENT_ENDPOINT')
            client.connect()
            client.barrier()
            if client.my_endpoint == self.server_endpoint:
                time.sleep(10)
                self.barrier_server.close_server()
        return local_path

    def download_from_bos(self,
                          fs_yaml=None,
                          local_path="./",
                          shard_num=-1,
                          shard_id=-1,
                          process_num=10):
        def multi_download(bos_path,
                           local_path,
                           filelist,
                           process_num=process_num):
            def _subprocess_download(files):
                for ff in files:
                    os.system("wget -q -P {} --no-check-certificate {}/{}".
                              format(local_path, bos_path, ff))
                    cmd = "tar -xf {}/{} -C {}".format(local_path, ff,
                                                       local_path)
                    os.system(cmd)

            dir_per_process = len(filelist) / process_num

            procs = []
            for i in range(process_num):
                process_filelist = filelist[i::process_num]
                p = multiprocessing.Process(
                    target=_subprocess_download, args=(process_filelist, ))
                procs.append(p)
                p.start()

            for proc in procs:
                proc.join()

        if is_first_worker():
            if not os.path.exists(local_path):
                os.system('mkdir {}'.format(local_path))
        yaml_file = fs_yaml.split('/')[-1]
        if not os.path.exists(yaml_file):
            if fs_yaml == None:
                raise Exception(
                    "Error: you should provide a yaml to download data from bos, you can find yaml examples in the following links:"
                )
            if is_first_worker():
                os.system("wget -q --no-check-certificate {}".format(fs_yaml))
                if not os.path.exists(yaml_file):
                    raise Exception(
                        "Error: If you provide a url, please check if your url is valid and is able to access; otherwise, please check if the yaml file is exists in your local path."
                    )
            else:
                while not os.path.exists(yaml_file):
                    time.sleep(1)

        _, ext = os.path.splitext(fs_yaml)
        assert ext in ['.yml', '.yaml'], "only support yaml files for now"
        with open(yaml_file) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        if 'bos_path' in cfg:
            bos_path = cfg["bos_path"]
        else:
            raise Exception("ERROR: Please figure your data path in BOS.")
        if is_first_worker():
            try:
                os.system(
                    "wget -q -P {} --no-check-certificate {}/filelist.txt".
                    format(local_path, bos_path))
                os.system("wget -q -P {} --no-check-certificate {}/meta.txt".
                          format(local_path, bos_path))
            except:
                raise Exception(
                    "ERROR: Your data dir should include filelist.txt and meta.txt"
                )
            with open('{}/meta.txt'.format(local_path), 'r') as fin:
                for line in fin:
                    current_file = line[:-1]
                    os.system("wget -q -P {} --no-check-certificate {}/{}".
                              format(local_path, bos_path, current_file))
        if shard_num > 0:
            assert (
                shard_id >= 0,
                "Please provide worker index by fleet.worker_index() if you want to download sharded data on each machine"
            )
            self.filelist = get_file_shard(shard_id, shard_num, local_path)
            need_download = check_exists(self.filelist, local_path)
            if need_download:
                multi_download(bos_path, local_path, self.filelist)
        else:
            if is_first_worker():
                self.filelist = get_file_shard(0, 1, local_path)
                need_download = check_exists(self.filelist, local_path)
                if need_download:
                    multi_download(bos_path, local_path, self.filelist)
        if self.need_barrier:
            client = BarrierClient()
            client.server_endpoint = self.server_endpoint
            client.my_endpoint = os.environ.get('PADDLE_CURRENT_ENDPOINT')
            client.connect()
            client.barrier()
            if client.my_endpoint == self.server_endpoint:
                time.sleep(10)
                self.barrier_server.close_server()
        return local_path
