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

from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient

def hdfs_ls(path, fs_name, ugi):
    configs = {
        "fs.default.name": fs_name,
        "hadoop.job.ugi": ugi,
    }
    hdfs_client = HDFSClient("$HADOOP_HOME", configs)
    filelist = []
    for i in path:
        cur_path = hdfs_client.ls(i)
        if fs_name.startswith("hdfs:"):
            cur_path = ["hdfs:" + j for j in cur_path]
        elif fs_name.startswith("afs:"):
            cur_path = ["afs:" + j for j in cur_path]
        filelist += cur_path
    return filelist

def hdfs_rmr(remote_path, fs_name, ugi):
    os.system("$HADOOP_HOME/bin/hadoop fs -Dhadoop.job.ugi={} -Dfs.default.name={} "
              "-rmr {}".format(ugi, fs_name, remote_path))

def hdfs_put(input, output, fs_name, ugi):
    os.system("$HADOOP_HOME/bin/hadoop fs -Dhadoop.job.ugi={} -Dfs.default.name={} "
              "-put {} {} &".format(ugi, fs_name, input, output))
