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
"""Plato Reader."""

from collections import namedtuple

import numpy as np

from knover.data.dialog_reader import DialogReader
from knover.utils import mask, pad_batch_data


class PlatoReader(DialogReader):
    """The implement of PlatoReader"""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = DialogReader.add_cmdline_args(parser)
        group.add_argument("--neg_pool_size", type=int, default=2 ** 16,
                           help="The size of random negative pool.")
        return group

    def __init__(self, args):
        super(PlatoReader, self).__init__(args)
        self.use_bow = args.use_bow
        self.use_nsp = args.use_nsp
        self.neg_pool_size = args.neg_pool_size
        if self.use_nsp:
            self.fields.extend(["neg_token_ids", "neg_type_ids", "neg_pos_ids"])
            if self.use_role:
                self.fields.append("neg_role_ids")
            if self.use_turn:
                self.fields.append("neg_turn_ids")
            self.Record = namedtuple("Record", self.fields, defaults=(None,) * len(self.fields))
            self.sort_key = lambda record: [2 * len(record.token_ids), len(record.neg_token_ids)]
        return

    def _mix_negative_sample(self, reader, neg_pool_size=2 ** 16):
        """Mix random negative samples into dataset."""
        def _gen_from_pool(pool):
            """Generate negative sample related fields from pool."""
            num_samples = len(pool)
            if num_samples == 1:
                # it is impossible to generate negative sample when the pool has only one sample
                return
            self.global_rng.shuffle(pool)
            for i in range(num_samples):
                j = (i + 1) % num_samples
                idx_i = pool[i].tgt_start_idx
                idx_j = pool[j].tgt_start_idx
                # add negative sample fields
                neg_fields = {}
                neg_fields["neg_token_ids"] = pool[i].token_ids[:idx_i] + pool[j].token_ids[idx_j:]
                neg_fields["neg_type_ids"] = pool[i].type_ids[:idx_i] + pool[j].type_ids[idx_j:]
                if self.position_style == "continuous":
                    neg_fields["neg_pos_ids"] = list(range(len(neg_fields["neg_token_ids"])))
                else:
                    neg_fields["neg_pos_ids"] = pool[i].pos_ids[:idx_i] + pool[j].pos_ids[idx_j:]
                if self.use_role:
                    neg_fields["neg_role_ids"] = pool[i].role_ids[:idx_i] + pool[j].role_ids[idx_j:]
                if self.use_turn:
                    neg_fields["neg_turn_ids"] = pool[i].turn_ids[:idx_i] + pool[j].turn_ids[idx_j:]
                pool[i] = pool[i]._replace(**neg_fields)
            self.global_rng.shuffle(pool)
            for record in pool:
                yield record

        def __wrapper__():
            pool = []
            for record in reader():
                pool.append(record)
                if len(pool) == neg_pool_size:
                    for record in _gen_from_pool(pool):
                        yield record
                    pool = []
            if len(pool) > 0:
                for record in _gen_from_pool(pool):
                    yield record
        return __wrapper__

    def _batch_reader(self, reader, phase=None, is_infer=False):
        """Construct a batch reader from a record reader."""
        if self.use_nsp and not is_infer:
            reader = self._mix_negative_sample(reader, self.neg_pool_size)
        return super(PlatoReader, self)._batch_reader(
            reader,
            phase=phase,
            is_infer=is_infer)

    def _pad_batch_records(self, batch_records, is_infer, **kwargs):
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

        batch_size = len(batch_token_ids)

        # padding
        batch["token_ids"] = pad_batch_data(batch_token_ids, pad_id=self.pad_id)
        batch["type_ids"] = pad_batch_data(batch_type_ids, pad_id=self.pad_id)
        batch["pos_ids"] = pad_batch_data(batch_pos_ids, pad_id=self.pad_id)
        if self.use_role:
            batch["role_ids"] = pad_batch_data(batch_role_ids, pad_id=self.pad_id)
        if self.use_turn:
            batch["turn_ids"] = pad_batch_data(batch_turn_ids, pad_id=self.pad_id)

        batch["generation_mask"] = self._gen_self_attn_mask(
            batch_token_ids,
            batch_tgt_start_idx=batch_tgt_start_idx,
            is_unidirectional=True,
            num_aux_token=1)
        if not is_infer:
            batch["recognition_mask"] = self._gen_self_attn_mask(
                batch_token_ids,
                is_unidirectional=False,
                num_aux_token=1)

            if self.use_nsp:
                batch_neg_token_ids = [record.neg_token_ids for record in batch_records]
                batch_neg_type_ids = [record.neg_type_ids for record in batch_records]
                batch_neg_pos_ids = [record.neg_pos_ids for record in batch_records]
                if self.use_role:
                    batch_neg_role_ids = [record.neg_role_ids for record in batch_records]
                if self.use_turn:
                    batch_neg_turn_ids = [record.neg_turn_ids for record in batch_records]

                batch["neg_token_ids"] = pad_batch_data(batch_neg_token_ids, pad_id=self.pad_id)
                batch["neg_type_ids"] = pad_batch_data(batch_neg_type_ids, pad_id=self.pad_id)
                batch["neg_pos_ids"] = pad_batch_data(batch_neg_pos_ids, pad_id=self.pad_id)
                if self.use_role:
                    batch["neg_role_ids"] = pad_batch_data(batch_neg_role_ids, pad_id=self.pad_id)
                if self.use_turn:
                    batch["neg_turn_ids"] = pad_batch_data(batch_neg_turn_ids, pad_id=self.pad_id)

                batch["neg_recognition_mask"] = self._gen_self_attn_mask(
                    batch_neg_token_ids,
                    is_unidirectional=False,
                    num_aux_token=1)

        if is_infer:
            tgt_ids = np.array([[[self.bos_id]]] * batch_size, dtype="int64")
            if self.position_style == "continuous":
                tgt_pos = np.array(batch_tgt_start_idx, dtype="int64")
            else:
                tgt_pos = np.zeros_like(batch_tgt_start_idx, dtype="int64")
            tgt_pos = tgt_pos.reshape(-1, 1, 1)
            batch["init_score"] = np.zeros_like(tgt_ids, dtype="float32").reshape(-1, 1).tolist()
            batch["tgt_ids"] = tgt_ids.tolist()
            batch["tgt_pos"] = tgt_pos.tolist()
            batch["parent_idx"] = np.array(range(batch_size), dtype="int32")
            batch["latent_id"] = np.zeros([batch_size], dtype="int32")

            batch["tgt_generation_mask"] = batch["generation_mask"][:, 0:1, :].astype("float32")

            batch_data_id = [record.data_id for record in batch_records]
            batch["data_id"] = np.array(batch_data_id).astype("int64").reshape([-1, 1])
        else:
            mask_return_list = mask(
                batch_tokens=batch_token_ids,
                vocab_size=self.vocab_size,
                tgt_starts=batch_tgt_start_idx,
                is_unidirectional=True,
                use_latent=True,
                use_bow=self.use_bow)
            batch["tgt_label"] = mask_return_list[0]
            batch["tgt_idx"] = mask_return_list[1]
            if self.use_bow:
                batch["bow_label"] = mask_return_list[2]
                batch["bow_idx"] = mask_return_list[3]

        return batch
