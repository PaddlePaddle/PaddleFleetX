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
import multiprocessing
import yaml
import os


class Downloader(object):
    def __init__(self, fs_yaml):
        pass


class ImageNetDownloader(Downloader):
    def __init__(self, fs_yaml):
        super(ImageNetDownloader, self).__init__(fs_yaml)
        _, ext = os.path.splitext(fs_yaml)
        assert ext in ['.yml', '.yaml'], "only support yaml files for now"
        with open(fs_yaml) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        if "hadoop_home" in cfg:
            self.hadoop_home = cfg["hadoop_home"]

            if "fs.default.name" in cfg and "hadoop.job.ugi" in cfg:
                self.hdfs_configs = {
                    "fs.default.name": cfg["fs.default.name"],
                    "hadoop.job.ugi": cfg["hadoop.job.ugi"]
                }

        if "imagenet_path" in cfg:
            self.default_path = cfg["imagenet_path"]
        else:
            print("WARNING: imagenet default path is empty")
        print(cfg)

    def download_from_hdfs(self, local_path="./", hdfs_path=None):
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
        multi_download(client, hdfs_path, local_path, 0, 1, 12)
        untar_files(local_path)
        return local_path

    def download_from_bos(self, bos_path):
        pass
