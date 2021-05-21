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
"""Sentence order prediction task."""

import math

from knover.core.task import Task
from knover.data.sop_reader import SOPReader
from knover.tasks import register_task


@register_task("SentenceOrderPrediction")
class SentenceOrderPrediction(Task):
    """Define sentence order prediction task."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = SOPReader.add_cmdline_args(parser)
        return group

    def __init__(self, args):
        super(SentenceOrderPrediction, self).__init__(args)
        self.reader = SOPReader(args)
        return

    def _post_process_infer_output(self, predictions):
        """Post-process inference output."""
        predictions = [{"data_id": data_id.tolist()[0], "score": score.tolist()[1]}
                       for data_id, score in zip(predictions["data_id"], predictions["scores"])]
        return predictions
