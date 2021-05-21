#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Evaluation main program."""

import argparse
from collections import defaultdict
import json
import os
import subprocess
import time

import paddle
import paddle.distributed.fleet as fleet
import paddle.fluid as fluid

import knover.models as models
import knover.tasks as tasks
from knover.utils import check_cuda, parse_args, str2bool, Timer


def setup_args():
    """
    Setup arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_distributed", type=str2bool, default=False)
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--eval_file", type=str, required=True)

    parser.add_argument("--log_steps", type=int, default=10)

    models.add_cmdline_args(parser)
    tasks.add_cmdline_args(parser)

    args = parse_args(parser)
    args.load(args.config_path, "Model")
    print(json.dumps(args, indent=2))
    return args


def evaluate(args):
    """
    Evaluation main function.
    """
    if args.is_distributed:
        fleet.init(is_collective=True)

        dev_count = fluid.core.get_cuda_device_count()
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))
        phase = "distributed_test"
    else:
        dev_count = 1
        gpu_id = 0
        phase = "test"
    place = fluid.CUDAPlace(gpu_id)

    task = tasks.create_task(args)
    model = models.create_model(args, place)
    eval_generator = task.get_data_loader(
        model,
        input_file=args.eval_file,
        num_part=dev_count,
        part_id=gpu_id,
        phase=phase
    )

    # run evaluation
    timer = Timer()
    timer.start()
    step = 0
    outputs = None
    for step, data in enumerate(eval_generator(), 1):
        part_outputs = task.eval_step(model, data)
        outputs = task.merge_metrics_and_statistics(outputs, part_outputs)

        if step % args.log_steps == 0:
            metrics = task.get_metrics(outputs)
            print(f"\tstep {step}:" + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    if args.is_distributed:
        # merge evaluation outputs in distributed mode.
        part_file = os.path.join(args.save_path, f"evaluation_output.part_{gpu_id}")
        with open(part_file, "w") as fp:
            json.dump(outputs, fp, ensure_ascii=False)
        part_finish_file = os.path.join(args.save_path, f"evaluation_output.part_{gpu_id}.finish")
        with open(part_finish_file, "w"):
            pass

        if gpu_id == 0:
            part_files = f"evaluation_output.part_*.finish"
            while True:
                ret = subprocess.getoutput(f"find {args.save_path} -maxdepth 1 -name {part_files}")
                num_completed = len(ret.split("\n"))
                if num_completed != dev_count:
                    time.sleep(1)
                    continue
                outputs = None
                for dev_id in range(dev_count):
                    part_file = os.path.join(args.save_path, f"evaluation_output.part_{dev_id}")
                    with open(part_file, "r") as fp:
                        part_outputs = json.load(fp)
                        outputs = task.merge_metrics_and_statistics(outputs, part_outputs)
                break
            subprocess.getoutput("rm " + os.path.join(args.save_path, f"evaluation_output.part*"))

    if gpu_id == 0:
        metrics = task.get_metrics(outputs)
        print(f"[Evaluation] " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
    return


if __name__ == "__main__":
    paddle.enable_static()
    args = setup_args()
    check_cuda(True)
    evaluate(args)
