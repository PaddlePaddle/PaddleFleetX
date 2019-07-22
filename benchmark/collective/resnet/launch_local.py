from __future__ import print_function

import subprocess
import os
import sys
import argparse

default_envs = {
    "PADDLE_TRAINER_ENDPOINTS":
    "127.0.0.1:6170,127.0.0.1:6171,127.0.0.1:6172,127.0.0.1:6173,127.0.0.1:6174,127.0.0.1:6175,127.0.0.1:6176,127.0.0.1:6177",
    "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
    "PATH": os.getenv("PATH"),
    "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
    "PADDLE_TRAINERS_NUM": "1",
    "NCCL_DEBUG": "INFO",
    "GLOG_v": "0",
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_RETRY_CNT": "0",
    "PYTHONPATH": os.getenv("PYTHONPATH", ""),
}

DEFAULT_GPUS = 8


def start_procs(gpus, entrypoint, entrypoint_args, log_dir):
    procs = []
    log_fns = []
    os.system("mkdir -p %s" % log_dir)

    # update parent envs
    for k, v in os.environ.items():
        if k.startswith("FLAGS_") or k.startswith("NCCL_") or k.startswith(
                "GLOG_"):
            default_envs[k] = v

    node_trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    current_ip = os.getenv("POD_IP", "127.0.0.1")
    trainer_ips = os.getenv("PADDLE_TRAINERS", current_ip).split(",")
    num_nodes = len(trainer_ips)
    all_nodes_devices_endpoints = ""
    gpus = gpus.split(',')
    for index, n in enumerate(trainer_ips):
        for i, gpu in enumerate(gpus):
            if all_nodes_devices_endpoints:
                all_nodes_devices_endpoints += ","
            all_nodes_devices_endpoints += "%s:617%d" % (n, i + index * 4)
    nranks = num_nodes * len(gpus)

    for i, gpu in enumerate(gpus):
        curr_env = {}
        curr_env.update(default_envs)
        curr_env.update({
            "FLAGS_selected_gpus": "%d" % (int(gpu)),
            "PADDLE_TRAINER_ID": "%d" % (node_trainer_id * len(gpus) + i),
            "PADDLE_CURRENT_ENDPOINT": all_nodes_devices_endpoints.split(',')[
                node_trainer_id * len(gpus) + i],
            "PADDLE_TRAINERS_NUM": "%d" % nranks,
            "PADDLE_TRAINER_ENDPOINTS": all_nodes_devices_endpoints
        })

        print("starting process ", i, entrypoint, entrypoint_args, curr_env)
        fn = open("%s/workerlog.%d" % (log_dir, int(gpu)), "w")
        log_fns.append(fn)
        cmd = [sys.executable, "-u", entrypoint] + entrypoint_args
        procs.append(subprocess.Popen(cmd, stdout=fn, stderr=fn, env=curr_env))

    for gpu in gpus:
        try:
            procs[int(gpu)].communicate()
            procs[int(gpu)].terminate()
            log_fns[int(gpu)].close()
        except:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='''start paddle training using multi-process mode.
NOTE: your train program ***must*** run as distributed nccl2 mode,
see: http://www.paddlepaddle.org/documentation/docs/zh/1.4/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2-
And your train program must read environment variables below in order to let different
process init properly:
FLAGS_selected_gpus
PADDLE_TRAINER_ID
PADDLE_CURRENT_ENDPOINT
PADDLE_TRAINERS_NUM
PADDLE_TRAINER_ENDPOINTS
POD_IP (current node ip address, not needed for local training)
''')
    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        help='GPU devices to launch the MP training.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default="logs",
        help='Directory to store log for each process.')
    parser.add_argument(
        'entrypoint_script',
        type=str,
        help="The entrypoint script to be launched in parallel,"
        "followed by all the arguments for each process,"
        "e.g. train.py --lr 0.1")
    parser.add_argument('entrypoint_args', nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    # launch multiple training process
    start_procs(args.gpus, args.entrypoint_script, args.entrypoint_args,
                args.log_dir)


if __name__ == "__main__":
    main()
