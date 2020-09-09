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


class Downloader(object):
    def __init__(self):
        pass

    def _read_yaml(self, fs_yaml):
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
        return cfg

class ImageNetDownloader(Downloader):
    def __init__(self):
        super(ImageNetDownloader, self).__init__()

    def download_from_hdfs(self, fs_yaml, local_path="./", hdfs_path=None):
        if not is_first_worker():
            return local_path
        cfg = self._read_yaml(fs_yaml)

        if "imagenet_path" in cfg:
            self.default_path = cfg["imagenet_path"]
        else:
            print("WARNING: imagenet default path is empty")

        def untar_files(local_path, process_num=10):
            def _subprocess_untar(files):
                for ff in files:
                    if "shard" in ff and ff.endswith(".tar"):
                        cmd = "tar -xf {} -C {}".format(local_path + "/" + ff,
                                                        local_path)
                        os.system(cmd)

            filelist = os.listdir(local_path)
            full_filelist = [x for x in filelist]
            dir_per_process = len(full_filelist) / process_num

            procs = []
            for i in range(process_num):
                process_filelist = full_filelist[i::process_num]
                p = multiprocessing.Process(
                    target=_subprocess_untar, args=(process_filelist, ))
                procs.append(p)
                p.start()

            for proc in procs:
                proc.join()

            cmd = "tar -xf {}/val.tar".format(local_path)
            os.system(cmd)

        if hdfs_path == None:
            hdfs_path = self.default_path
        client = HDFSClient(self.hadoop_home, self.hdfs_configs)
        gpu_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
        num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 0))
        multi_download(client, hdfs_path, local_path, gpu_id, num_trainers, 12)
        untar_files(local_path)
        return local_path

    def download_from_bos(self, local_path="./"):
        if not is_first_worker():
            return local_path
        print("Start download data")
        os.system(
            'wget -q -P {} --no-check-certificate https://fleet.bj.bcebos.com/small_datasets/imagenet/val.txt'.
            format(local_path))
        os.system(
            'wget -q -P {} --no-check-certificate https://fleet.bj.bcebos.com/small_datasets/imagenet/train.txt'.
            format(local_path))
        os.system(
            'wget -q -P {} --no-check-certificate https://fleet.bj.bcebos.com/small_datasets/imagenet/val.tar.gz'.
            format(local_path))
        os.system('tar -xf {}/val.tar.gz -C {}'.format(local_path, local_path))

        def untar(target_file, steps):
            for i in steps:
                os.system(
                    'wget -q -P {} --no-check-certificate https://fleet.bj.bcebos.com/small_datasets/imagenet/shard{}.tar'.
                    format(target_file, i))
                os.system('tar -xf {}/shard{}.tar -C {}'.format(target_file, i,
                                                                target_file))

        set_lists = {}
        for process in range(20):
            set_lists[process] = []
        for num in range(62):
            set_lists[num % 10].append(num)
        procs = []
        for i in range(10):
            p = multiprocessing.Process(
                target=untar, args=(
                    local_path,
                    set_lists[i], ))
            procs.append(p)
            p.start()

        for proc in procs:
            proc.join()
        return local_path


class WikiDataDownloader(Downloader):
    def __init__(self, language="cn", data_size="small"):
        super(WikiDataDownloader, self).__init__()
        assert(language in ["en", "cn"])
        assert(data_size in ["normal", "small"])
        self.language = language
        self.data_size = data_size
        self.wiki_path = None

    def download_from_hdfs(self, fs_yaml, local_path="./", hdfs_path=None):
        if not is_first_worker():
            time.sleep(3)
            return local_path
        cfg = self._read_yaml(fs_yaml)
        if "wiki_path" in cfg:
            self.wiki_path = cfg["wiki_path"]
        else:
            print("WARNING: wikipedia default path is empty")
        
        dirname = "{}_{}".format(self.language, self.data_size)
        if hdfs_path == None:
            hdfs_path = os.path.join(self.wiki_path, dirname)
        client = HDFSClient(self.hadoop_home, self.hdfs_configs)
        multi_download(client, hdfs_path, local_path, 0, 1, 2)
        if dirname == "cn_small":
            os.system('tar -xf {}/train_data.tar.gz -C {}'.format(local_path,
                                                              local_path))            
        return local_path

    def download_from_bos(self, local_path):
        if not is_first_worker():
            time.sleep(3)
            return local_path
        print("Start download data")
        dirname = "{}_{}".format(self.language, self.data_size)
        if dirname == "cn_small":
            os.system(
                'wget -q -P {} --no-check-certificate https://fleet.bj.bcebos.com/small_datasets/wiki/cn_small/vocab.txt'.
                format(local_path))
            os.system(
                'wget -q -P {} --no-check-certificate https://fleet.bj.bcebos.com/small_datasets/wiki/cn_small/train_data.tar.gz'.
                format(local_path))
            os.system('tar -xf {}/train_data.tar.gz -C {}'.format(local_path,
                                                              local_path))
        elif dirname == "en_small":
            os.system(
                'wget -q -P {} --no-check-certificate https://fleet.bj.bcebos.com/small_datasets/wiki/en_small/part-9-9.txt'.
                format(local_path))
        else:
            raise ValueError("this type of dataset is not supported: "
                 "language: {}, data_size: {}".format(self.language, self.data_size))
        return local_path

class WMTDataDownloader(Downloader):
    def __init__(self):
        super(WMTDataDownloader, self).__init__()

    def download_from_hdfs(self, fs_yaml, local_path="./", hdfs_path=None):
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

        if "wmt_path" in cfg:
            self.default_path = cfg["wmt_path"]
        else:
            print("WARNING: imagenet default path is empty")

        if hdfs_path == None:
            hdfs_path = self.default_path
        client = HDFSClient(self.hadoop_home, self.hdfs_configs)
        gpu_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
        num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 0))
        multi_download(client, hdfs_path, local_path, gpu_id, num_trainers, 12)
        return local_path

    def download_from_bos(self, local_path='./'):
        gpu_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
        if gpu_id != 0:
            time.sleep(3)
            return local_path
        print("Start download data")
        os.system(
            'wget -q -P {} --no-check-certificate https://fleet.bj.bcebos.com/small_datasets/wmt/vocab_all.bpe.32000'.
            format(local_path))
        os.system(
            'wget -q -P {} --no-check-certificate https://fleet.bj.bcebos.com/small_datasets/wmt/train.tok.clean.bpe.32000.en-de'.
            format(local_path))

        return local_path
