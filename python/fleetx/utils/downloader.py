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
from paddle.fluid.contrib.utils import HDFSClient, multi_download
import time
from .env import is_first_worker
import multiprocessing
import yaml
import os


def check_images_ready(local_path):
    while True:
        with open("{}/train.txt".format(local_path)) as fin:
            filelist = []
            ready_list = []
            for line in fin:
                current_image = line.split(' ')
                filelist.append(current_image)
                image_path = "{}/train/{}".format(local_path, current_image[0])
                if os.path.exists(image_path):
                    ready_list.append(image_path)
            if len(filelist) == len(ready_list):
                return
            else:
                time.sleep(3)


def check_exists(local_path):
    if not os.path.exists("{}/data_info.txt".format(local_path)):
        return True
    with open("{}/data_info.txt".format(local_path)) as fin:
        for line in fin:
            current_file = line[:-1]
            if not os.path.exists("{}/{}".format(local_path, current_file)):
                print("{}/{}".format(local_path, current_file))
                return True
        return False


def untar_files_with_check(local_path, trainer_id, trainer_num,
                           process_num=10):
    print("Waiting others to finish download......")
    while True:
        if os.path.exists("{}/data_info.txt".format(local_path)):
            filelist = []
            ready_filelist = []
            with open("{}/data_info.txt".format(local_path)) as fin:
                for line in fin:
                    filelist.append(line[:-1])
                for ff in filelist:
                    if os.path.exists("{}/{}".format(local_path, ff)):
                        ready_filelist.append("{}/{}".format(local_path, ff))
                if len(ready_filelist) == len(filelist):
                    num_per_trainer = int(len(ready_filelist) /
                                          trainer_num) + 1
                    if (trainer_id + 1
                        ) * num_per_trainer < len(ready_filelist):
                        sub_list = ready_filelist[trainer_id * num_per_trainer:
                                                  (trainer_id + 1
                                                   ) * num_per_trainer]
                        print(sub_list)
                        return sub_list
                    else:
                        sub_list = ready_filelist[trainer_id *
                                                  num_per_trainer:]
                        print(sub_list)
                        return sub_list
                else:
                    time.sleep(2)
        else:
            time.sleep(2)


class Downloader(object):
    def __init__(self):
        pass

    def download_from_hdfs(self, fs_yaml=None, local_path="./"):
        """
        Download from hdfs
        The configurations are configured in fs_yaml file:

        TODO: add example and yaml argument fields introduction

        """
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

        def untar_files(local_path, tar_list, process_num=10):
            def _subprocess_untar(files):
                for ff in files:
                    if ff.endswith(".tar"):
                        cmd = "tar -xf {} -C {}".format(ff, local_path)
                        os.system(cmd)

            dir_per_process = len(tar_list) / process_num

            procs = []
            for i in range(process_num):
                process_filelist = tar_list[i::process_num]
                p = multiprocessing.Process(
                    target=_subprocess_untar, args=(process_filelist, ))
                procs.append(p)
                p.start()

            for proc in procs:
                proc.join()

        if hdfs_path == None:
            hdfs_path = self.default_path
        client = HDFSClient(self.hadoop_home, self.hdfs_configs)
        PADDLE_TRAINER_ENDPOINTS = os.environ.get('PADDLE_TRAINER_ENDPOINTS')
        endpoints = PADDLE_TRAINER_ENDPOINTS.split(",")
        current_endpoint = os.environ.get('PADDLE_CURRENT_ENDPOINT')
        need_download = check_exists(local_path)
        if need_download:
            multi_download(client, hdfs_path, local_path,
                           endpoints.index(current_endpoint),
                           len(endpoints), 12)
        tar_list = untar_files_with_check(local_path,
                                          endpoints.index(current_endpoint),
                                          len(endpoints))
        if os.path.exists("{}/train".format(local_path)):
            print(
                "Warning: You may already have imagenet dataset in {}, please check!".
                format(local_path))
        untar_files(local_path, tar_list)
        check_images_ready(local_path)
        return local_path

    def download_from_bos(self, fs_yaml=None, local_path="./"):
        pass

