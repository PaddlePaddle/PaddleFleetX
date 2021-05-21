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
"""SOP Reader."""

from collections import namedtuple

import numpy as np

from knover.data.dialog_reader import DialogReader
from knover.utils import mask, pad_batch_data, str2bool


class SOPReader(DialogReader):
    """SOP Reader."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = DialogReader.add_cmdline_args(parser)
        return group

    def __init__(self, args):
        super(SOPReader, self).__init__(args)
        self.fields.append("label")
        self.Record = namedtuple("Record", self.fields, defaults=(None,) * len(self.fields))
        return

    def _convert_example_to_record(self, example, is_infer):
        """Convert example to record which can be used as the model's input."""
        record = super(SOPReader, self)._convert_example_to_record(example, False)
        if "label" in example._fields:
            record = record._replace(label=int(example.label))
        return record

    def _mix_negative_sample(self, reader):
        """Mix negative samples into dataset."""
        def __wrapper__():
            for record in reader():
                yield record._replace(label=1)

                field_values = {}
                tgt_idx = record.tgt_start_idx
                last_idx = 1
                for i in range(tgt_idx - 1):
                    if record.token_ids[i] == self.eos_id:
                        last_idx = i + 1

                src_token_ids = record.token_ids[:last_idx] + record.token_ids[tgt_idx + 1:]
                tgt_token_ids = [self.bos_id] + record.token_ids[last_idx:tgt_idx]
                field_values["token_ids"] = src_token_ids + tgt_token_ids
                field_values["type_ids"] = [0] * len(src_token_ids) + [1] * len(tgt_token_ids)

                if self.position_style == "relative":
                    ctx_len = len(src_token_ids)
                    src_pos_ids = [
                        self.max_tgt_len + ctx_len - i - 1
                        for i in range(ctx_len)
                    ]
                    tgt_pos_ids = list(range(len(tgt_token_ids)))
                    field_values["pos_ids"] = src_pos_ids + tgt_pos_ids
                else:
                    field_values["pos_ids"] = record.pos_ids

                if self.use_role:
                    field_values["role_ids"] = (
                        record.role_ids[:last_idx]
                        + [record.role_ids[last_idx]] * len(record.token_ids[tgt_idx + 1:])
                        + [0] * len(tgt_token_ids)
                    )

                if self.use_turn:
                    field_values["turn_ids"] = (
                        record.turn_ids[:last_idx]
                        + [record.turn_ids[last_idx]] * len(record.token_ids[tgt_idx + 1:])
                        + [0] * len(tgt_token_ids)
                    )

                for k in field_values:
                    assert len(field_values[k]) == len(field_values["token_ids"])

                neg_record = self.Record(
                    **field_values,
                    tgt_start_idx=len(src_token_ids),
                    data_id=-1,
                    label=0
                )
                if len(src_token_ids) < self.max_src_len:
                    # filter negative sample which source sequence is too long
                    yield neg_record
        return __wrapper__

    def _batch_reader(self, reader, phase=None, is_infer=False):
        """Construct a batch reader from a record reader."""
        return super(SOPReader, self)._batch_reader(
            self._mix_negative_sample(reader),
            phase=phase,
            is_infer=is_infer)

    def _pad_batch_records(self, batch_records, is_infer, phase=None):
        """Padding a batch of records and construct model's inputs."""
        batch = {}
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_type_ids = [record.type_ids for record in batch_records]
        batch_pos_ids = [record.pos_ids for record in batch_records]
        if self.use_role:
            batch_role_ids = [record.role_ids for record in batch_records]
        if self.use_turn:
            batch_turn_ids = [record.turn_ids for record in batch_records]
        batch_tgt_start_idx = [record.tgt_start_idx for record in batch_records]
        batch_label = [record.label for record in batch_records]

        batch_mask_token_ids, tgt_label, tgt_idx, label_idx = mask(
            batch_tokens=batch_token_ids,
            vocab_size=self.vocab_size,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            mask_id=self.mask_id,
            tgt_starts=batch_tgt_start_idx,
            labels=batch_label,
            is_unidirectional=False)
        if not is_infer:
            # use masking token ids in training
            batch_token_ids = batch_mask_token_ids
        batch["token_ids"] = pad_batch_data(batch_token_ids, pad_id=self.pad_id)
        batch["type_ids"] = pad_batch_data(batch_type_ids, pad_id=self.pad_id)
        batch["pos_ids"] = pad_batch_data(batch_pos_ids, pad_id=self.pad_id)
        if self.use_role:
            batch["role_ids"] = pad_batch_data(batch_role_ids, pad_id=self.pad_id)
        if self.use_turn:
            batch["turn_ids"] = pad_batch_data(batch_turn_ids, pad_id=self.pad_id)
        attention_mask = self._gen_self_attn_mask(batch_token_ids, is_unidirectional=False)

        batch["attention_mask"] = attention_mask
        batch["label_idx"] = label_idx

        if not is_infer:
            batch_label = np.array(batch_label).astype("int64").reshape([-1, 1])
            batch["label"] = batch_label
            batch["tgt_label"] = tgt_label
            batch["tgt_idx"] = tgt_idx
        else:
            batch_data_id = [record.data_id for record in batch_records]
            batch["data_id"] = np.array(batch_data_id).astype("int64").reshape([-1, 1])

        return batch
