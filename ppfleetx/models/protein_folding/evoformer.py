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

from ppfleetx.distributed.protein_folding import bp, dap

from .attentions import (
    MSARowAttentionWithPairBias,
    MSAColumnGlobalAttention,
    MSAColumnAttention,
    TriangleMultiplication,
    TriangleAttention, )

from .common import (
    Transition,
    Dropout,
    recompute_wrapper,
    dgram_from_positions, )

from .template import (TemplateEmbedding, )
from .outer_product_mean import (OuterProductMean, )

from . import (
    residue_constants,
    all_atom, )


class EvoformerIteration(nn.Layer):
    """Single iteration (block) of Evoformer stack.

    Jumper et al. (2021) Suppl. Alg. 6 "EvoformerStack" lines 2-10
    """

    def __init__(self, channel_num, config, global_config, is_extra_msa=False):
        super(EvoformerIteration, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.is_extra_msa = is_extra_msa

        assert self.global_config.outer_product_mean_position in [
            'origin', 'middle', 'first', 'end'
        ]

        # Row-wise Gated Self-attention with Pair Bias
        self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(
            channel_num, self.config.msa_row_attention_with_pair_bias,
            self.global_config, is_extra_msa)
        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.msa_row_attention_with_pair_bias)
        self.msa_row_attn_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        if self.is_extra_msa:
            self.msa_column_global_attention = MSAColumnGlobalAttention(
                channel_num, config.msa_column_attention, global_config)
            dropout_rate, dropout_axis = self._parse_dropout_params(
                self.msa_column_global_attention)
            self.msa_col_attn_dropout = nn.Dropout(
                dropout_rate, axis=dropout_axis) \
                    if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)
        else:
            self.msa_column_attention = MSAColumnAttention(
                channel_num, config.msa_column_attention, global_config)
            dropout_rate, dropout_axis = self._parse_dropout_params(
                self.msa_column_attention)
            self.msa_col_attn_dropout = nn.Dropout(
                dropout_rate, axis=dropout_axis) \
                    if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.msa_transition = Transition(
            channel_num, self.config.msa_transition, self.global_config,
            is_extra_msa, 'msa_transition')
        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.msa_transition)
        self.msa_transition_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
                if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        # OuterProductMean
        self.outer_product_mean = OuterProductMean(
            channel_num,
            self.config.outer_product_mean,
            self.global_config,
            self.is_extra_msa,
            name='outer_product_mean')

        # Dropout
        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.outer_product_mean)
        self.outer_product_mean_dropout = nn.Dropout(
            dropout_rate, axis=dropout_axis) \
                if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        # Triangle Multiplication.
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            channel_num,
            self.config.triangle_multiplication_outgoing,
            self.global_config,
            name='triangle_multiplication_outgoing')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_outgoing)
        self.triangle_outgoing_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_multiplication_incoming = TriangleMultiplication(
            channel_num,
            self.config.triangle_multiplication_incoming,
            self.global_config,
            name='triangle_multiplication_incoming')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_multiplication_incoming)
        self.triangle_incoming_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        # TriangleAttention.
        self.triangle_attention_starting_node = TriangleAttention(
            channel_num,
            self.config.triangle_attention_starting_node,
            self.global_config,
            name='triangle_attention_starting_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_starting_node)
        self.triangle_starting_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        self.triangle_attention_ending_node = TriangleAttention(
            channel_num,
            self.config.triangle_attention_ending_node,
            self.global_config,
            name='triangle_attention_ending_node')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.triangle_attention_ending_node)
        self.triangle_ending_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

        # Pair transition.
        self.pair_transition = Transition(
            channel_num, self.config.pair_transition, self.global_config,
            is_extra_msa, 'pair_transition')

        dropout_rate, dropout_axis = self._parse_dropout_params(
            self.pair_transition)
        self.pair_transition_dropout = nn.Dropout(dropout_rate, axis=dropout_axis) \
            if not self.global_config.use_dropout_nd else Dropout(dropout_rate, axis=dropout_axis)

    def _parse_dropout_params(self, module):
        dropout_rate = 0.0 if self.global_config.deterministic else \
            module.config.dropout_rate
        dropout_axis = None
        if module.config.shared_dropout:
            dropout_axis = {
                'per_row': [0, 2, 3],
                'per_column': [0, 1, 3],
            }[module.config.orientation]

        return dropout_rate, dropout_axis

    def outer_product_mean_origin(self, msa_act, pair_act, masks):

        assert bp.get_world_size(
        ) == 1, "Branch Parallel degree must be 1 for outer_product_mean_origin"

        msa_mask, pair_mask = masks['msa'], masks['pair']

        # [B, N_seq//dap_size, N_res, c_m]
        residual = self.msa_row_attention_with_pair_bias(msa_act, msa_mask,
                                                         pair_act)
        residual = self.msa_row_attn_dropout(residual)
        msa_act = msa_act + residual

        # [B, N_seq//dap_size, N_res, c_m] => [B, N_seq, N_res//dap_size, c_m]
        msa_act = dap.row_to_col(msa_act)

        if self.is_extra_msa:
            # [B, N_seq, N_res//dap_size, c_m]
            residual = self.msa_column_global_attention(msa_act, msa_mask)
            residual = self.msa_col_attn_dropout(residual)
            msa_act = msa_act + residual

            # [B, N_seq, N_res//dap_size, c_m]
            residual = self.msa_transition(msa_act, msa_mask)
            residual = self.msa_transition_dropout(residual)
            msa_act = msa_act + residual

        else:
            # [B, N_seq, N_res//dap_size, c_m]
            residual = self.msa_column_attention(msa_act, msa_mask)
            residual = self.msa_col_attn_dropout(residual)
            msa_act = msa_act + residual

            # [B, N_seq, N_res//dap_size, c_m]
            residual = self.msa_transition(msa_act, msa_mask)
            residual = self.msa_transition_dropout(residual)
            msa_act = msa_act + residual

        # [B, N_res//dap_size, N_res, c_z]
        residual = self.outer_product_mean(msa_act, msa_mask)
        outer_product_mean = self.outer_product_mean_dropout(residual)
        pair_act = pair_act + outer_product_mean

        # [B, N_seq, N_res//dap_size, c_m] => [B, N_seq//dap_size, N_res, c_m]
        msa_act = dap.col_to_row(msa_act)

        # scatter if using dap, otherwise do nothing
        pair_mask_row = dap.scatter(pair_mask, axis=1)
        pair_mask_col = dap.scatter(pair_mask, axis=2)

        # [B, N_res//dap_size, N_res, c_z]
        # TODO(GuoxiaWang): why have diffrence whether remove pair_act = pair_act.clone()
        # pair_act = pair_act.clone()
        residual = self.triangle_multiplication_outgoing(pair_act,
                                                         pair_mask_row)
        residual = self.triangle_outgoing_dropout(residual)
        pair_act = pair_act + residual

        # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res//dap_size, c_z]
        pair_act = dap.row_to_col(pair_act)
        # [B, N_res, N_res//dap_size, c_z]
        residual = self.triangle_multiplication_incoming(pair_act,
                                                         pair_mask_col)
        residual = self.triangle_incoming_dropout(residual)
        pair_act = pair_act + residual

        # [B, N_res, N_res//dap_size, c_z] => [B, N_res//dap_size, N_res, c_z]
        pair_act = dap.col_to_row(pair_act)
        # [B, N_res//dap_size, N_res, c_z]
        residual = self.triangle_attention_starting_node(pair_act,
                                                         pair_mask_row)
        residual = self.triangle_starting_dropout(residual)
        pair_act = pair_act + residual

        # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res//dap_size, c_z]
        pair_act = dap.row_to_col(pair_act)
        # [B, N_res, N_res//dap_size, c_z]
        residual = self.triangle_attention_ending_node(pair_act, pair_mask_col)
        residual = self.triangle_ending_dropout(residual)
        pair_act = pair_act + residual

        residual = self.pair_transition(pair_act, pair_mask)
        residual = self.pair_transition_dropout(residual)
        pair_act = pair_act + residual

        # [B, N_res, N_res//dap_size, c_z] => [B, N_res//dap_size, N_res, c_z]
        pair_act = dap.col_to_row(pair_act)

        return msa_act, pair_act

    def outer_product_mean_first(self, msa_act, pair_act, masks):
        raise NotImplementedError(
            "BP or DAP does not support outer_product_mean_first")

    def outer_product_mean_end(self, msa_act, pair_act, masks):
        msa_mask, pair_mask = masks['msa'], masks['pair']

        if bp.get_world_size() > 1:
            # Note(GuoxiaWang): add zeros trigger the status of stop_gradient=False within recompute context.
            pair_act = pair_act + paddle.zeros_like(pair_act)

            # Note(GuoxiaWang): reduce the pair_act's gradient from msa branch and pair branch
            if not pair_act.stop_gradient:
                pair_act._register_grad_hook(bp.all_reduce)

            if bp.get_rank_in_group() == 0:
                # [B, N_seq//dap_size, N_res, c_m]
                residual = self.msa_row_attention_with_pair_bias(
                    msa_act, msa_mask, pair_act)
                residual = self.msa_row_attn_dropout(residual)
                msa_act = msa_act + residual

                # [B, N_seq//dap_size, N_res, c_m] => [B, N_seq, N_res//dap_size, c_m]
                msa_act = dap.row_to_col(msa_act)

                if self.is_extra_msa:
                    # [B, N_seq, N_res//dap_size, c_m]
                    residual = self.msa_column_global_attention(msa_act,
                                                                msa_mask)
                    residual = self.msa_col_attn_dropout(residual)
                    msa_act = msa_act + residual

                    # [B, N_seq, N_res//dap_size, c_m]
                    residual = self.msa_transition(msa_act, msa_mask)
                    residual = self.msa_transition_dropout(residual)
                    msa_act = msa_act + residual

                else:
                    # [B, N_seq, N_res//dap_size, c_m]
                    residual = self.msa_column_attention(msa_act, msa_mask)
                    residual = self.msa_col_attn_dropout(residual)
                    msa_act = msa_act + residual

                    # [B, N_seq, N_res//dap_size, c_m]
                    residual = self.msa_transition(msa_act, msa_mask)
                    residual = self.msa_transition_dropout(residual)
                    msa_act = msa_act + residual

                # [B, N_res//dap_size, N_res, c_z]
                residual = self.outer_product_mean(msa_act, msa_mask)
                outer_product_mean = self.outer_product_mean_dropout(residual)

                # [B, N_seq, N_res//dap_size, c_m] => [B, N_seq//dap_size, N_res, c_m]
                msa_act = dap.col_to_row(msa_act)

            if bp.get_rank_in_group() == 1:
                # scatter if using dap, otherwise do nothing
                pair_mask_row = dap.scatter(pair_mask, axis=1)
                pair_mask_col = dap.scatter(pair_mask, axis=2)

                # [B, N_res//dap_size, N_res, c_z]
                residual = self.triangle_multiplication_outgoing(pair_act,
                                                                 pair_mask_row)
                residual = self.triangle_outgoing_dropout(residual)
                pair_act = pair_act + residual

                # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res//dap_size, c_z]
                pair_act = dap.row_to_col(pair_act)
                # [B, N_res, N_res//dap_size, c_z]
                residual = self.triangle_multiplication_incoming(pair_act,
                                                                 pair_mask_col)
                residual = self.triangle_incoming_dropout(residual)
                pair_act = pair_act + residual

                # [B, N_res, N_res//dap_size, c_z] => [B, N_res//dap_size, N_res, c_z]
                pair_act = dap.col_to_row(pair_act)
                # [B, N_res//dap_size, N_res, c_z]
                residual = self.triangle_attention_starting_node(pair_act,
                                                                 pair_mask_row)
                residual = self.triangle_starting_dropout(residual)
                pair_act = pair_act + residual

                # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res//dap_size, c_z]
                pair_act = dap.row_to_col(pair_act)
                # [B, N_res, N_res//dap_size, c_z]
                residual = self.triangle_attention_ending_node(pair_act,
                                                               pair_mask_col)
                residual = self.triangle_ending_dropout(residual)
                pair_act = pair_act + residual

                residual = self.pair_transition(pair_act, pair_mask)
                residual = self.pair_transition_dropout(residual)
                pair_act = pair_act + residual

                # [B, N_res, N_res//dap_size, c_z] => [B, N_res//dap_size, N_res, c_z]
                pair_act = dap.col_to_row(pair_act)

                outer_product_mean = paddle.zeros_like(pair_act)
                outer_product_mean.stop_gradient = pair_act.stop_gradient

            # TODO(GuoxiaWang): fix PyLayer ctx illegal access
            msa_act = paddle.assign(msa_act)
            pair_act = paddle.assign(pair_act)

            msa_act, pair_act = bp.sync_evoformer_results(outer_product_mean,
                                                          msa_act, pair_act)
            # TODO(GuoxiaWang): fix PyLayer ctx illegal access
            pair_act = paddle.assign(pair_act)
            return msa_act, pair_act

        else:
            # [B, N_seq//dap_size, N_res, c_m]
            residual = self.msa_row_attention_with_pair_bias(msa_act, msa_mask,
                                                             pair_act)
            residual = self.msa_row_attn_dropout(residual)
            msa_act = msa_act + residual

            # [B, N_seq//dap_size, N_res, c_m] => [B, N_seq, N_res//dap_size, c_m]
            msa_act = dap.row_to_col(msa_act)

            if self.is_extra_msa:
                # [B, N_seq, N_res//dap_size, c_m]
                residual = self.msa_column_global_attention(msa_act, msa_mask)
                residual = self.msa_col_attn_dropout(residual)
                msa_act = msa_act + residual

                # [B, N_seq, N_res//dap_size, c_m]
                residual = self.msa_transition(msa_act, msa_mask)
                residual = self.msa_transition_dropout(residual)
                msa_act = msa_act + residual

            else:
                # [B, N_seq, N_res//dap_size, c_m]
                residual = self.msa_column_attention(msa_act, msa_mask)
                residual = self.msa_col_attn_dropout(residual)
                msa_act = msa_act + residual

                # [B, N_seq, N_res//dap_size, c_m]
                residual = self.msa_transition(msa_act, msa_mask)
                residual = self.msa_transition_dropout(residual)
                msa_act = msa_act + residual

            # [B, N_res//dap_size, N_res, c_z]
            residual = self.outer_product_mean(msa_act, msa_mask)
            outer_product_mean = self.outer_product_mean_dropout(residual)

            # [B, N_seq, N_res//dap_size, c_m] => [B, N_seq//dap_size, N_res, c_m]
            msa_act = dap.col_to_row(msa_act)

            # scatter if using dap, otherwise do nothing
            pair_mask_row = dap.scatter(pair_mask, axis=1)
            pair_mask_col = dap.scatter(pair_mask, axis=2)

            # [B, N_res//dap_size, N_res, c_z]
            # TODO(GuoxiaWang): why have diffrence whether remove pair_act = pair_act.clone()
            # pair_act = pair_act.clone()
            residual = self.triangle_multiplication_outgoing(pair_act,
                                                             pair_mask_row)
            residual = self.triangle_outgoing_dropout(residual)
            pair_act = pair_act + residual

            # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res//dap_size, c_z]
            pair_act = dap.row_to_col(pair_act)
            # [B, N_res, N_res//dap_size, c_z]
            residual = self.triangle_multiplication_incoming(pair_act,
                                                             pair_mask_col)
            residual = self.triangle_incoming_dropout(residual)
            pair_act = pair_act + residual

            # [B, N_res, N_res//dap_size, c_z] => [B, N_res//dap_size, N_res, c_z]
            pair_act = dap.col_to_row(pair_act)
            # [B, N_res//dap_size, N_res, c_z]
            residual = self.triangle_attention_starting_node(pair_act,
                                                             pair_mask_row)
            residual = self.triangle_starting_dropout(residual)
            pair_act = pair_act + residual

            # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res//dap_size, c_z]
            pair_act = dap.row_to_col(pair_act)
            # [B, N_res, N_res//dap_size, c_z]
            residual = self.triangle_attention_ending_node(pair_act,
                                                           pair_mask_col)
            residual = self.triangle_ending_dropout(residual)
            pair_act = pair_act + residual

            residual = self.pair_transition(pair_act, pair_mask)
            residual = self.pair_transition_dropout(residual)
            pair_act = pair_act + residual

            # [B, N_res, N_res//dap_size, c_z] => [B, N_res//dap_size, N_res, c_z]
            pair_act = dap.col_to_row(pair_act)

            pair_act = pair_act + outer_product_mean

            return msa_act, pair_act

    def forward(self, msa_act, pair_act, masks):

        if self.global_config.outer_product_mean_position in [
                'origin', 'middle'
        ]:
            msa_act, pair_act = self.outer_product_mean_origin(msa_act,
                                                               pair_act, masks)

        elif self.global_config.outer_product_mean_position == 'first':
            msa_act, pair_act = self.outer_product_mean_first(msa_act,
                                                              pair_act, masks)

        elif self.global_config.outer_product_mean_position == 'end':
            msa_act, pair_act = self.outer_product_mean_end(msa_act, pair_act,
                                                            masks)

        else:
            raise Error(
                "Only support outer_product_mean_position in ['origin', 'middle', ''first', 'end'] now!"
            )

        return msa_act, pair_act


class DistEmbeddingsAndEvoformer(nn.Layer):
    """Embeds the input data and runs Evoformer.

    Produces the MSA, single and pair representations.
    Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5-18
    """

    def __init__(self, channel_num, config, global_config):
        super(DistEmbeddingsAndEvoformer, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        # InputEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5
        # Jumper et al. (2021) Suppl. Alg. 3 "InputEmbedder"
        self.preprocess_1d = nn.Linear(
            channel_num['target_feat'],
            self.config.msa_channel,
            name='preprocess_1d')
        self.preprocess_msa = nn.Linear(
            channel_num['msa_feat'],
            self.config.msa_channel,
            name='preprocess_msa')
        self.left_single = nn.Linear(
            channel_num['target_feat'],
            self.config.pair_channel,
            name='left_single')
        self.right_single = nn.Linear(
            channel_num['target_feat'],
            self.config.pair_channel,
            name='right_single')

        # RecyclingEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
        # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"
        if self.config.recycle_pos:
            self.prev_pos_linear = nn.Linear(self.config.prev_pos.num_bins,
                                             self.config.pair_channel)

        # RelPosEmbedder
        # Jumper et al. (2021) Suppl. Alg. 4 "relpos"
        # Jumper et al. (2021) Suppl. Alg. 5 "one_hot"
        if self.config.max_relative_feature:
            self.pair_activiations = nn.Linear(
                2 * self.config.max_relative_feature + 1,
                self.config.pair_channel)

        if self.config.recycle_features:
            self.prev_msa_first_row_norm = nn.LayerNorm(
                self.config.msa_channel)
            self.prev_pair_norm = nn.LayerNorm(self.config.pair_channel)

        # Embed templates into the pair activations.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-13
        if self.config.template.enabled:
            self.channel_num['template_angle'] = 57
            self.channel_num['template_pair'] = 88
            self.template_embedding = TemplateEmbedding(
                self.channel_num, self.config.template, self.global_config)

        # ExtraMSAEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 14-16
        self.extra_msa_activations = nn.Linear(
            25,  # 23 (20aa+unknown+gap+mask) + 1 (has_del) + 1 (del_val)
            self.config.extra_msa_channel)

        # Extra MSA Stack.
        # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
        self.extra_msa_stack = nn.LayerList()
        for _ in range(self.config.extra_msa_stack_num_block):
            self.extra_msa_stack.append(
                EvoformerIteration(
                    self.channel_num,
                    self.config.evoformer,
                    self.global_config,
                    is_extra_msa=True))

        # Embed templates torsion angles
        if self.config.template.enabled and self.config.template.embed_torsion_angles:
            c = self.config.msa_channel
            self.template_single_embedding = nn.Linear(
                self.channel_num['template_angle'], c)
            self.template_projection = nn.Linear(c, c)

        # Main trunk of the network
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 17-18
        self.evoformer_iteration = nn.LayerList()
        for _ in range(self.config.evoformer_num_block):
            self.evoformer_iteration.append(
                EvoformerIteration(
                    self.channel_num,
                    self.config.evoformer,
                    self.global_config,
                    is_extra_msa=False))

        self.single_activations = nn.Linear(self.config.msa_channel,
                                            self.config.seq_channel)

    def _pseudo_beta_fn(self, aatype, all_atom_positions, all_atom_masks):
        gly_id = paddle.ones_like(aatype) * residue_constants.restype_order[
            'G']
        is_gly = paddle.equal(aatype, gly_id)

        ca_idx = residue_constants.atom_order['CA']
        cb_idx = residue_constants.atom_order['CB']

        n = len(all_atom_positions.shape)
        pseudo_beta = paddle.where(
            paddle.tile(
                paddle.unsqueeze(
                    is_gly, axis=-1), [1] * len(is_gly.shape) + [3]),
            paddle.squeeze(
                all_atom_positions.slice([n - 2], [ca_idx], [ca_idx + 1]),
                axis=-2),
            paddle.squeeze(
                all_atom_positions.slice([n - 2], [cb_idx], [cb_idx + 1]),
                axis=-2))

        if all_atom_masks is not None:
            m = len(all_atom_masks)
            pseudo_beta_mask = paddle.where(
                is_gly,
                paddle.squeeze(
                    all_atom_masks.slice([m - 1], [ca_idx], [ca_idx + 1]),
                    axis=-1),
                paddle.squeeze(
                    all_atom_masks.slice([m - 1], [cb_idx], [cb_idx + 1]),
                    axis=-1))
            pseudo_beta_mask = paddle.squeeze(pseudo_beta_mask, axis=-1)
            return pseudo_beta, pseudo_beta_mask
        else:
            return pseudo_beta

    def _create_extra_msa_feature(self, batch):
        # 23: 20aa + unknown + gap + bert mask
        msa_1hot = nn.functional.one_hot(batch['extra_msa'], 23)
        msa_feat = [
            msa_1hot, paddle.unsqueeze(
                batch['extra_has_deletion'], axis=-1), paddle.unsqueeze(
                    batch['extra_deletion_value'], axis=-1)
        ]
        return paddle.concat(msa_feat, axis=-1)

    def forward(self, batch):
        # InputEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 5
        # Jumper et al. (2021) Suppl. Alg. 3 "InputEmbedder"
        preprocess_1d = self.preprocess_1d(batch['target_feat'])
        # preprocess_msa = self.preprocess_msa(batch['msa_feat'])
        msa_activations = paddle.unsqueeze(preprocess_1d, axis=1) + \
                    self.preprocess_msa(batch['msa_feat'])

        right_single = self.right_single(
            batch['target_feat'])  # 1, n_res, 22 -> 1, n_res, 128
        right_single = paddle.unsqueeze(
            right_single, axis=1)  # 1, n_res, 128 -> 1, 1, n_res, 128
        left_single = self.left_single(
            batch['target_feat'])  # 1, n_res, 22 -> 1, n_res, 128
        left_single = paddle.unsqueeze(
            left_single, axis=2)  # 1, n_res, 128 -> 1, n_res, 1, 128
        pair_activations = left_single + right_single

        mask_2d = paddle.unsqueeze(
            batch['seq_mask'], axis=1) * paddle.unsqueeze(
                batch['seq_mask'], axis=2)

        # Inject previous outputs for recycling.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
        # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"
        if self.config.recycle_pos and 'prev_pos' in batch:
            prev_pseudo_beta = self._pseudo_beta_fn(batch['aatype'],
                                                    batch['prev_pos'], None)
            dgram = dgram_from_positions(prev_pseudo_beta,
                                         **self.config.prev_pos)
            pair_activations += self.prev_pos_linear(dgram)

        if self.config.recycle_features:
            if 'prev_msa_first_row' in batch:
                prev_msa_first_row = self.prev_msa_first_row_norm(batch[
                    'prev_msa_first_row'])

                # A workaround for `jax.ops.index_add`
                msa_first_row = paddle.squeeze(
                    msa_activations[:, 0, :], axis=1)
                msa_first_row += prev_msa_first_row
                msa_first_row = paddle.unsqueeze(msa_first_row, axis=1)
                msa_activations = paddle.concat(
                    [msa_first_row, msa_activations[:, 1:, :]], axis=1)

            if 'prev_pair' in batch:
                pair_activations += self.prev_pair_norm(batch['prev_pair'])

        # RelPosEmbedder
        # Jumper et al. (2021) Suppl. Alg. 4 "relpos"
        # Jumper et al. (2021) Suppl. Alg. 5 "one_hot"
        if self.config.max_relative_feature:
            pos = batch['residue_index']  # [bs, N_res]
            offset = paddle.unsqueeze(pos, axis=[-1]) - \
                paddle.unsqueeze(pos, axis=[-2])
            rel_pos = nn.functional.one_hot(
                paddle.clip(
                    offset + self.config.max_relative_feature,
                    min=0,
                    max=2 * self.config.max_relative_feature),
                2 * self.config.max_relative_feature + 1)
            rel_pos_bias = self.pair_activiations(rel_pos)
            pair_activations += rel_pos_bias

        # TemplateEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-13
        if self.config.template.enabled:
            template_batch = {
                k: batch[k]
                for k in batch if k.startswith('template_')
            }
            template_pair_repr = self.template_embedding(
                pair_activations, template_batch, mask_2d)
            pair_activations += template_pair_repr

        # ExtraMSAEmbedder
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 14-16
        extra_msa_feat = self._create_extra_msa_feature(batch)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)

        # ==================================================
        #  Extra MSA Stack
        # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
        # ==================================================
        extra_msa_stack_input = {
            'msa': extra_msa_activations,
            'pair': pair_activations,
        }

        if bp.get_world_size() > 1:
            extra_msa_stack_input['msa'] = bp.broadcast_grad_for_backward(
                extra_msa_stack_input['msa'], 0)

        # scatter if using dap, otherwise do nothing
        # [B, N_seq, N_res, c_m] => [B, N_seq//dap_size, N_res, c_m]
        extra_msa_stack_input['msa'] = dap.scatter(
            extra_msa_stack_input['msa'], axis=1)
        # [B, N_res, N_res, c_z] => [B, N_res//dap_size, N_res, c_z]
        extra_msa_stack_input['pair'] = dap.scatter(
            extra_msa_stack_input['pair'], axis=1)

        for idx, extra_msa_stack_iteration in enumerate(self.extra_msa_stack):
            extra_msa_act, extra_pair_act = recompute_wrapper(
                extra_msa_stack_iteration,
                extra_msa_stack_input['msa'],
                extra_msa_stack_input['pair'],
                {'msa': batch['extra_msa_mask'],
                 'pair': mask_2d},
                is_recompute=self.training and
                idx >= self.config.extra_msa_stack_recompute_start_block_index)
            extra_msa_stack_output = {
                'msa': extra_msa_act,
                'pair': extra_pair_act
            }
            extra_msa_stack_input = {
                'msa': extra_msa_stack_output['msa'],
                'pair': extra_msa_stack_output['pair']
            }

        # gather if using dap, otherwise do nothing
        # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res, c_z]
        extra_msa_stack_output['pair'] = dap.gather(
            extra_msa_stack_output['pair'], axis=1)

        evoformer_input = {
            'msa': msa_activations,
            'pair': extra_msa_stack_output['pair'],
        }

        evoformer_masks = {
            'msa': batch['msa_mask'],
            'pair': mask_2d,
        }

        # ==================================================
        #  Template angle feat
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 7-8
        # ==================================================
        if self.config.template.enabled and self.config.template.embed_torsion_angles:
            num_templ, num_res = batch['template_aatype'].shape[1:]

            aatype_one_hot = nn.functional.one_hot(batch['template_aatype'],
                                                   22)
            # Embed the templates aatype, torsion angles and masks.
            # Shape (templates, residues, msa_channels)
            ret = all_atom.atom37_to_torsion_angles(
                aatype=batch['template_aatype'],
                all_atom_pos=batch['template_all_atom_positions'],
                all_atom_mask=batch['template_all_atom_masks'],
                # Ensure consistent behaviour during testing:
                placeholder_for_undefined=not self.global_config.zero_init)

            template_features = paddle.concat(
                [
                    aatype_one_hot,
                    paddle.reshape(ret['torsion_angles_sin_cos'],
                                   [-1, num_templ, num_res, 14]),
                    paddle.reshape(ret['alt_torsion_angles_sin_cos'],
                                   [-1, num_templ, num_res, 14]),
                    ret['torsion_angles_mask']
                ],
                axis=-1)

            template_activations = self.template_single_embedding(
                template_features)
            template_activations = nn.functional.relu(template_activations)
            template_activations = self.template_projection(
                template_activations)

            # Concatenate the templates to the msa.
            evoformer_input['msa'] = paddle.concat(
                [evoformer_input['msa'], template_activations], axis=1)

            # Concatenate templates masks to the msa masks.
            # Use mask from the psi angle, as it only depends on the backbone atoms
            # from a single residue.
            torsion_angle_mask = ret['torsion_angles_mask'][..., 2]
            torsion_angle_mask = torsion_angle_mask.astype(evoformer_masks[
                'msa'].dtype)
            evoformer_masks['msa'] = paddle.concat(
                [evoformer_masks['msa'], torsion_angle_mask], axis=1)

        if bp.get_world_size() > 1:
            evoformer_input['msa'] = bp.broadcast_grad_for_backward(
                evoformer_input['msa'], 0)

        # scatter if using dap, otherwise do nothing
        # [B, N_seq, N_res, c_m] => [B, N_seq//dap_size, N_res, c_m]
        evoformer_input['msa'] = dap.scatter(evoformer_input['msa'], axis=1)
        # [B, N_res, N_res, c_z] => [B, N_res//dap_size, N_res, c_z]
        evoformer_input['pair'] = dap.scatter(evoformer_input['pair'], axis=1)

        # ==================================================
        #  Main MSA Stack
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 17-18
        # ==================================================
        for idx, evoformer_block in enumerate(self.evoformer_iteration):
            msa_act, pair_act = recompute_wrapper(
                evoformer_block,
                evoformer_input['msa'],
                evoformer_input['pair'],
                evoformer_masks,
                is_recompute=self.training and
                idx >= self.config.evoformer_recompute_start_block_index)
            evoformer_output = {'msa': msa_act, 'pair': pair_act}
            evoformer_input = {
                'msa': evoformer_output['msa'],
                'pair': evoformer_output['pair'],
            }

        # gather if using dap, otherwise do nothing
        # [B, N_seq//dap_size, N_res, c_m] => [B, N_seq, N_res, c_m]
        evoformer_output['msa'] = dap.gather(evoformer_output['msa'], axis=1)
        # [B, N_res//dap_size, N_res, c_z] => [B, N_res, N_res, c_z]
        evoformer_output['pair'] = dap.gather(evoformer_output['pair'], axis=1)

        msa_activations = evoformer_output['msa']
        pair_activations = evoformer_output['pair']
        single_activations = self.single_activations(msa_activations[:, 0])

        num_seq = batch['msa_feat'].shape[1]
        output = {
            'single': single_activations,
            'pair': pair_activations,
            # Crop away template rows such that they are not used
            # in MaskedMsaHead.
            'msa': msa_activations[:, :num_seq],
            'msa_first_row': msa_activations[:, 0],
        }

        return output
