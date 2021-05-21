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
"""Convert generation result into readable text file."""

import argparse
import json
import sys


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--json_file", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    with open(args.in_file) as in_f, open(args.pred_file) as pred_f, open(args.out_file, "w") as out_f, open(args.json_file, "w") as json_f:
        sys.stdout = out_f
        headers = next(in_f).strip().split("\t")
        no = 0
        results = []
        for in_line, pred_line in zip(in_f, pred_f):
            no += 1
            cols = in_line.strip().split("\t")
            example = dict(zip(headers, cols))
            pred = pred_line.strip()
            print(f"Case {no}")
            print("Context:")
            for s in example["src"].split(" [SEP] "):
                print(f"\t{s}")
            print("Prediction:")
            print(f"\t{pred}")
            print()
            result = {
                "src": example["src"].split(" [SEP] "),
                "pred": pred
            }
            results.append(result)
        json.dump(results, json_f, indent=2)


if __name__ == "__main__":
    args = setup_args()
    main(args)
