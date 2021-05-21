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
"""Compare multiple generation results in excel.

Convert multiple json-format generation result files into a readable xlsx file.
"""

import argparse
import json
import os
import sys

import xlsxwriter


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

    workbook = xlsxwriter.Workbook(args.out_file)
    worksheet = workbook.add_worksheet("Details")

    cell_format = workbook.add_format()
    cell_format.set_text_wrap()

    green_format = workbook.add_format()
    green_format.set_bg_color("#D8E4BC")
    green_format.set_text_wrap()

    red_format = workbook.add_format()
    red_format.set_bg_color("#E6B8B7")
    red_format.set_text_wrap()

    blue_format = workbook.add_format()
    blue_format.set_bg_color("#C5D9F1")
    blue_format.set_text_wrap()

    row = 0
    def write_row(row_content, cell_format=cell_format):
        nonlocal row
        worksheet.write_row(row, 0, row_content, cell_format=cell_format)
        row += 1
        return


    if args.in_style == "static":
        worksheet.set_column(0, 0, 16)
        worksheet.set_column(1, 1, 80)
        worksheet.set_column(2, 4, 16)
        worksheet.freeze_panes(1, 0)
        write_row(["", "", "Coherence", "Informativeness", "Engagingness"])
    else:
        worksheet.set_column(0, 0, 16)
        col = 1
        message_list = [""]
        for _ in range(len(args.labels.split(","))):
            worksheet.set_column(col, col, 60)
            worksheet.set_column(col + 1, col + 2, 16)
            message_list.extend(["", "Coherence", "Informativeness"])
            col += 3
        worksheet.freeze_panes(1, 1)
        write_row(message_list)

    for no, example_result_list in enumerate(zip(*results_list), 1):
        write_row([f"Case {no}"])
        if args.in_style == "static":
            write_row(["Context:"])
            for s in example_result_list[0]["src"]:
                write_row(["", s.strip()])
            write_row(["Prediction:"])
            for label, result in zip(args.labels.split(","), example_result_list):
                write_row([f"{label}:", result["pred"]])
            write_row([])
        else:
            message_list = ["Model name:"]
            for label in args.labels.split(","):
                message_list.extend([label, "", ""])
            write_row(message_list)
            for i, turn_list in enumerate(zip(*example_result_list)):
                turn_list = list(turn_list)
                message_list = []
                if i == 0:
                    message_list.append("[Start]:")
                elif i % 2 == 1:
                    message_list.append("[Bot1]:")
                else:
                    message_list.append("[Bot2]:")
                for utt in turn_list:
                    message_list.extend([utt, "", ""])
                if i == 0:
                    write_row(message_list, green_format)
                elif i % 2 == 1:
                    write_row(message_list, red_format)
                else:
                    write_row(message_list, blue_format)
            write_row(["Engagingness"])
            write_row(["Humanness"])
            write_row([])
    workbook.close()


if __name__ == "__main__":
    args = setup_args()
    main(args)
