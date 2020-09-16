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
from paddle.distributed.fleet.base.util_factory import fleet_util
import hashlib
from .env import is_first_worker, get_node_info
import multiprocessing
import yaml
import os


def barrier():
    fleet.init(is_collective=True)
    role = fleet._role_maker
    fleet_util._set_role_maker(role)
    fleet_util.barrier(comm_world='worker')


def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_exists(filelist, local_path):
    with open("{}/filelist.txt".format(local_path)) as fin:
        for line in fin:
            current_file = line.split(' ')[0]
            current_md5 = line.split(' ')[1][:-1]
            if current_file in filelist:
                if (not os.path.exists("{}/{}".format(
                        local_path, current_file))) or get_md5("{}/{}".format(
                            local_path, current_file)) != current_md5:
                    os.system("rm -rf {}/*".format(local_path))
                    return True
        return False


def get_file_shard(node_id, node_num, local_path):
    full_list = []
    with open("{}/filelist.txt".format(local_path), 'rb') as fin:
        for line in fin:
            full_list.append(line.split(' ')[0])
    return full_list[node_id::node_num]


class Downloader(object):
    def __init__(self):
        pass

    def download_from_hdfs(self, fs_yaml=None, local_path="./", sharded=True):
        """
        Download from hdfs
        The configurations are configured in fs_yaml file:
        TODO: add example and yaml argument fields introduction
        """
        role = fleet.base.role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        fleet_util._set_role_maker(role)
        if not is_first_worker():
            fleet_util.barrier()
            return local_path
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

        def multi_download(client,
                           hdfs_path,
                           local_path,
                           filelist,
                           process_num=10):
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

        client = HDFSClient(self.hadoop_home, self.hdfs_configs)
        client.download('{}/filelist.txt'.format(hdfs_path),
                        '{}/filelist.txt'.format(local_path))
        client.download('{}/meta.txt'.format(hdfs_path),
                        '{}/meta.txt'.format(local_path))
        with open('{}/meta.txt'.format(local_path), 'rb') as fin:
            for line in fin:
                current_file = line[:-1]
                client.download('{}/{}'.format(hdfs_path, current_file),
                                '{}/{}'.format(local_path, current_file))

        if sharded:
            node_id, node_num = get_node_info()
        else:
            node_id, node_num = 0, 1
        self.filelist = get_file_shard(node_id, node_num, local_path)
        need_download = check_exists(self.filelist, local_path)
        if need_download:
            multi_download(client, hdfs_path, local_path, self.filelist)
        fleet_util.barrier()
        return local_path

    def download_from_bos(self, fs_yaml=None, local_path="./", sharded=True):
        role = fleet.base.role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        fleet_util._set_role_maker(role)
        if fs_yaml == None:
            print("Error: you should provide a yaml to download data from bos")
            print("you can find yaml examples in the following links: ")
        if not is_first_worker():
            fleet_util.barrier()
        _, ext = os.path.splitext(fs_yaml)
        assert ext in ['.yml', '.yaml'], "only support yaml files for now"
        with open(fs_yaml) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        if 'data_path' in cfg:
            bos_path = cfg["bos_path"]

        def multi_download(bos_path, local_path, filelist, process_num=10):
            def _subprocess_download(files):
                for ff in files:
                    os.system("wget -q -P {} --no-check-certificate {}/{}".
                              format(local_path, bos_path, ff))
                    cmd = "tar -xf {} -C {}".format(ff, local_path)
                    os.system(cmd)

            dir_per_process = len(tar_list) / process_num

            procs = []
            for i in range(process_num):
                process_filelist = tar_list[i::process_num]
                p = multiprocessing.Process(
                    target=_subprocess_download, args=(process_filelist, ))
                procs.append(p)
                p.start()

            for proc in procs:
                proc.join()

        os.system("wget -q -P {} --no-check-certificate {}/filelist.txt".
                  format(local_path, bos_path))
        os.system("wget -q -P {} --no-check-certificate {}/meta.txt".format(
            local_path, bos_path))
        with open('{}/meta.txt'.format(local_path), 'rb') as fin:
            for line in fin:
                current_file = line[:-1]
                os.system("wget -q -P {} --no-check-certificate {}/{}".format(
                    local_path, bos_path, current_file))
        if sharded:
            node_id, node_num = get_node_info()
        else:
            node_id, node_num = 0, 1
        self.filelist = get_file_shard(node_id, node_num, local_path)
        need_download = check_exists(self.filelist, local_path)
        if need_download:
            multi_download(bos_path, local_path, self.filelist)
        fleet_util.barrier()
        return local_path
