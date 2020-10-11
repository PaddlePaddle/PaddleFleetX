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

import sys
import os
import yaml
from argparse import ArgumentParser, REMAINDER


class Submitter(object):
    def __init__(self):
        pass


class PaddleCloudSubmitter(Submitter):
    def __init__(self):
        super(PaddleCloudSubmitter, self).__init__()
        self.submit_str = "paddlecloud job " \
                          "--server {} " \
                          "--port {} " \
                          "train --job-version paddle-fluid-custom " \
                          "--image-addr {} " \
                          "--cluster-name {} " \
                          "--group-name {} " \
                          "--k8s-gpu-cards {} " \
                          "--k8s-priority high " \
                          "--k8s-wall-time 00:00:00 " \
                          "--k8s-memory 350Gi " \
                          "--job-name {} " \
                          "--start-cmd 'sh start_job.sh' " \
                          "--job-conf config.ini " \
                          "--k8s-trainers {} {} " \
                          "--k8s-cpu-cores 35 " \
                          "--file-dir {} " \
                          "--is-auto-over-sell {}"

    def get_start_job(self, yml_cfg):
        # set up config.ini
        log_fs_name = yml_cfg['log_fs_name']
        log_fs_ugi = yml_cfg['log_fs_ugi']
        log_output_path = yml_cfg['log_output_path']
        if 'afs' in log_fs_name:
            storage_type = 'afs'
        config = "storage_type = \"{}\"\n".format(storage_type)
        config += "fs_name = \"{}\"\n".format(log_fs_name)
        config += "fs_ugi = \"{}\"\n".format(log_fs_ugi)
        config += "output_path = \"{}\"\n".format(log_output_path)
        config += "FLAGS_rpc_deadline=3000000\n"
        config += "NCCL_DEBUG=INFO\n"
        with open("config.ini", "w") as fout:
            fout.write(config)
        os.system("cat config.ini")
        # by default, we use baidu pip source
        pip_src = "--index-url=http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com"
        if "pip_src" in yml_cfg:
            pip_src = yml_cfg["pip_src"]
        if "whl_install_commands" in yml_cfg:
            whl_install_commands = yml_cfg['whl_install_commands']
        if "commands" not in yml_cfg:
            print("ERROR: you should specify training or inference command")
            exit(0)
        commands = yml_cfg["commands"]
        job_sh = ""
        job_sh += "unset http_proxy\nunset https_proxy\n"
        job_sh += "pip uninstall paddlepaddle -y\n"
        job_sh += "pip uninstall paddlepaddle-gpu -y\n"
        job_sh += "pip uninstall fleet-x -y\n"
        for whl_cmd in whl_install_commands:
            job_sh += "{}\n".format(whl_cmd)
        for cmd in commands:
            job_sh += "{}\n".format(cmd)
        with open("start_job.sh", "w") as fout:
            fout.write(job_sh)

    def submit(self, yml):
        with open(yml) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        assert "server" in cfg, "server has to be configured"
        server = cfg["server"]
        port = "80"
        assert "num_trainers" in cfg, "num_trainers should be configured"
        num_trainers = cfg["num_trainers"]
        assert "num_cards" in cfg, "num_cards should be configured"
        num_cards = cfg["num_cards"]
        if "job_prefix" not in cfg:
            job_prefix = "paddle_cloud_test"
        else:
            job_prefix = cfg["job_prefix"]
        assert "image_addr" in cfg, "image_addr should be configured"
        image_addr = cfg["image_addr"]
        assert "cluster_name" in cfg, "cluster_name should be configured"
        cluster_name = cfg["cluster_name"]
        assert "group_name" in cfg, "group_name should be configured"
        group_name = cfg["group_name"]
        self.get_start_job(cfg)
        file_dir = "./"
        if "file_dir" in cfg:
            file_dir = cfg["file_dir"]
        if "over_sell" in cfg:
            over_sell = cfg['over_sell']
        else:
            over_sell = 1
        distribute_suffix = " --k8s-not-local --distribute-job-type NCCL2" \
                            if int(num_trainers) > 1 else ""
        pcloud_submit_cmd = self.submit_str.format(
            server, port, image_addr, cluster_name, group_name, num_cards,
            "{}_N{}C{}".format(job_prefix, num_trainers, num_cards),
            num_trainers, distribute_suffix, file_dir, over_sell)
        print(pcloud_submit_cmd)
        os.system(pcloud_submit_cmd)


def _parse_args():
    parser = ArgumentParser('''submit paddlecloud jobs''')
    parser.add_argument(
        "-f", type=str, default="", help="set up your job in a yaml file")
    return parser.parse_args()


def submitter():
    args = _parse_args()
    submitter = PaddleCloudSubmitter()
    if args.f == "":
        print("You should specify a yaml file for cloud submission")
        print("fleetsub -f cloud.yaml")
        exit(0)
    submitter.submit(args.f)


if __name__ == "__main__":
    submitter()
