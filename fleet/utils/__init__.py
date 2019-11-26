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
import hdfs
import system_info
hdfs_ls=hdfs.hdfs_ls
hdfs_rmr=hdfs.hdfs_rmr
hdfs_put=hdfs.hdfs_put
launch_system_monitor=system_info.launch_system_monitor
get_system_info=system_info.get_system_info
get_monitor_result=system_info.get_monitor_result

