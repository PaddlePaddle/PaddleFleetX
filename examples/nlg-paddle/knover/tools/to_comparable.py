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
"""Compare multiple generation results.

Convert multiple json-format generation result files into a readable tsv file.
"""

import argparse
import json
import os
import sys


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_files", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--in_style", type=str, default="static", choices=["static", "self_chat"])
    args = parser.parse_args()
    return args


def main(args):
    results_list = []
    for json_file in args.json_files.split(","):
        with open(json_file) as json_f:
            results = json.load(json_f)
            results_list.append(results)

    with open(args.out_file, "w") as out_f:
        sys.stdout = out_f
        for no, example_result_list in enumerate(zip(*results_list), 1):
            print(f"Case {no}")
            if args.in_style == "static":
                print("Context:")
                for s in example_result_list[0]["src"]:
                    print(f"\t{s.strip()}")
                print("Prediction:")
                for label, result in zip(args.labels.split(","), example_result_list):
                    print(f"\t{label}:\t{result['pred']}")
                print()
            else:
                print("Model name:\t" + "\t".join(args.labels.split(",")))
                for i, turn_list in enumerate(zip(*example_result_list)):
                    if i == 0:
                        print("[Start]:\t" + "\t".join(turn_list))
                    elif i % 2 == 1:
                        print("[Bot1]:\t" + "\t".join(turn_list))
                    else:
                        print("[Bot2]:\t" + "\t".join(turn_list))
                print()


if __name__ == "__main__":
    args = setup_args()
    main(args)
