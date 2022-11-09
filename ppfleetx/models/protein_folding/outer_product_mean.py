#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn

from ppfleetx.distributed.protein_folding import dap

from .common import subbatch


class OuterProductMean(nn.Layer):
    """Computes mean outer product.

    Jumper et al. (2021) Suppl. Alg. 10 "OuterProductMean"
    """

    def __init__(self,
                 channel_num,
                 config,
                 global_config,
                 is_extra_msa,
                 name='outer_product_mean'):
        super(OuterProductMean, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        if is_extra_msa:
            c_m = channel_num['extra_msa_channel']
        else:
            c_m = channel_num['msa_channel']

        self.layer_norm_input = nn.LayerNorm(c_m, name='layer_norm_input')
        self.left_projection = nn.Linear(
            c_m, self.config.num_outer_channel, name='left_projection')
        self.right_projection = nn.Linear(
            c_m, self.config.num_outer_channel, name='right_projection')

        if self.global_config.zero_init:
            init_w = nn.initializer.Constant(value=0.0)
        else:
            init_w = nn.initializer.KaimingNormal()

        self.output_w = paddle.create_parameter(
            [
                self.config.num_outer_channel, self.config.num_outer_channel,
                channel_num['pair_channel']
            ],
            'float32',
            default_initializer=init_w)
        self.output_b = paddle.create_parameter(
            [channel_num['pair_channel']],
            'float32',
            default_initializer=nn.initializer.Constant(value=0.0))

    def forward(self, act, mask):
        """Builds OuterProductMean module.

        Arguments:
        act: MSA representation, shape [batch, N_seq, N_res, c_m].
        mask: MSA mask, shape [batch, N_seq, N_res].

        Returns:
        Update to pair representation, shape [batch, N_res, N_res, c_z].
        """
        # [B, N_seq, N_res//dap_size, c_m]
        act = self.layer_norm_input(act)
        # [B, N_seq, N_res//dap_size, c_m] => [B, N_seq, N_res//dap_size, num_outer_channel]
        right_act_before = self.right_projection(act)
        # [B, N_seq, N_res//dap_size, num_outer_channel] => [B, N_seq, N_res, num_outer_channel]
        right_act = dap.all_gather(right_act_before, axis=2)

        # [B, N_seq, N_res//dap_size, c_m] => [B, N_seq, N_res//dap_size, num_outer_channel]
        left_act = self.left_projection(act)
        # [B, N_seq, N_res] => [B, N_seq, N_res, 1]
        mask = paddle.unsqueeze(mask, axis=-1)
        # [B, N_seq, N_res, 1] => [B, N_seq, N_res//dap_size, 1]
        mask_col = dap.scatter(mask, axis=2)
        left_act = mask_col * left_act

        # [B, N_seq, N_res//dap_size, 1], [B, N_seq, N_res, 1] => [B, N_res//dap_size, N_res, 1]
        epsilon = 1e-3
        norm = paddle.einsum('nabc,nadc->nbdc', mask_col, mask) + epsilon

        def compute_chunk(left_act, right_act):
            # This is equivalent to
            #
            # act = jnp.einsum('abc,ade->dceb', left_act, right_act)
            # act = jnp.einsum('dceb,cef->bdf', act, output_w) + output_b
            #
            # but faster. maybe for subbatch inference?

            # [B, N_seq, N_res//dap_size, num_outer_channel] => [B, N_seq, num_outer_channel, N_res//dap_size]
            left_act = left_act.transpose([0, 1, 3, 2])
            # wait if using async communication and dap, otherwise do nothing
            right_act_after = dap.all_gather_opp(right_act, axis=2)
            # [B, N_seq, num_outer_channel, N_res//dap_size], [B, N_seq, N_res, num_outer_channel]
            # => [B, N_res, num_outer_channel, num_outer_channel, N_res//dap_size]
            act = paddle.einsum('nacb,nade->ndceb', left_act, right_act_after)
            # [B, N_res, num_outer_channel, num_outer_channel, N_res//dap_size], [num_outer_channel, num_outer_channel, c_z]
            # => [B, N_res, N_res//dap_size, c_z]
            act = paddle.einsum('ndceb,cef->ndbf', act,
                                self.output_w) + self.output_b
            # [B, N_res, N_res//dap_size, c_z] => [B, N_res//dap_size, N_res, c_z]
            return act.transpose([0, 2, 1, 3])

        if not self.training:
            # low memory mode using subbatch
            sb_chunk = subbatch(compute_chunk, [0], [2],
                                self.config.chunk_size, 1)
            act = sb_chunk(left_act, right_act)
        else:
            act = compute_chunk(left_act, right_act)

        act = act / norm

        return act
