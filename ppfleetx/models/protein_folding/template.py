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

from .attentions import (
    Attention,
    TriangleMultiplication,
    TriangleAttention, )

from .common import (
    Transition,
    Dropout,
    recompute_wrapper,
    dgram_from_positions,
    subbatch, )

from . import (residue_constants, )
from . import (quat_affine, )


class TemplatePair(nn.Layer):
    """Pair processing for the templates.

    Jumper et al. (2021) Suppl. Alg. 16 "TemplatePairStack" lines 2-6
    """

    def __init__(self, channel_num, config, global_config):
        super(TemplatePair, self).__init__()
        self.config = config
        self.global_config = global_config

        channel_num = {}
        channel_num[
            'pair_channel'] = self.config.triangle_attention_ending_node.value_dim

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

        self.pair_transition = Transition(
            channel_num,
            self.config.pair_transition,
            self.global_config,
            is_extra_msa=False,
            transition_type='pair_transition')

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

    def forward(self, pair_act, pair_mask):
        """Builds one block of TemplatePair module.

        Arguments:
        pair_act: Pair activations for single template, shape [batch, N_res, N_res, c_t].
        pair_mask: Pair mask, shape [batch, N_res, N_res].

        Returns:
        Updated pair_act, shape [batch, N_res, N_res, c_t].
        """

        pair_mask_row = dap.scatter(pair_mask, axis=1)
        pair_mask_col = dap.scatter(pair_mask, axis=2)

        residual = self.triangle_attention_starting_node(pair_act,
                                                         pair_mask_row)
        residual = self.triangle_starting_dropout(residual)
        pair_act = pair_act + residual

        pair_act = dap.row_to_col(pair_act)
        residual = self.triangle_attention_ending_node(pair_act, pair_mask_col)
        residual = self.triangle_ending_dropout(residual)
        pair_act = pair_act + residual

        pair_act = dap.col_to_row(pair_act)
        residual = self.triangle_multiplication_outgoing(pair_act,
                                                         pair_mask_row)
        residual = self.triangle_outgoing_dropout(residual)
        pair_act = pair_act + residual

        pair_act = dap.row_to_col(pair_act)
        residual = self.triangle_multiplication_incoming(pair_act,
                                                         pair_mask_col)
        residual = self.triangle_incoming_dropout(residual)
        pair_act = pair_act + residual

        residual = self.pair_transition(pair_act, pair_mask)
        residual = self.pair_transition_dropout(residual)
        pair_act = pair_act + residual

        pair_act = dap.col_to_row(pair_act)

        return pair_act


class SingleTemplateEmbedding(nn.Layer):
    """Embeds a single template.

    Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9+11
    """

    def __init__(self, channel_num, config, global_config):
        super(SingleTemplateEmbedding, self).__init__()
        self.config = config
        self.channel_num = channel_num
        self.global_config = global_config

        self.embedding2d = nn.Linear(channel_num['template_pair'],
                                     self.config.template_pair_stack.
                                     triangle_attention_ending_node.value_dim)

        self.template_pair_stack = nn.LayerList()
        for _ in range(self.config.template_pair_stack.num_block):
            self.template_pair_stack.append(
                TemplatePair(self.channel_num, self.config.template_pair_stack,
                             self.global_config))

        self.output_layer_norm = nn.LayerNorm(self.config.attention.key_dim)

    def forward(self, query_embedding, batch, mask_2d):
        """Build the single template embedding.

        Arguments:
            query_embedding: Query pair representation, shape [batch, N_res, N_res, c_z].
            batch: A batch of template features (note the template dimension has been
                stripped out as this module only runs over a single template).
            mask_2d: Padding mask (Note: this doesn't care if a template exists,
                unlike the template_pseudo_beta_mask).

        Returns:
            A template embedding [N_res, N_res, c_z].
        """
        assert mask_2d.dtype == query_embedding.dtype
        dtype = query_embedding.dtype
        num_res = batch['template_aatype'].shape[1]
        template_mask = batch['template_pseudo_beta_mask']
        template_mask_2d = template_mask[..., None] * template_mask[...,
                                                                    None, :]
        template_mask_2d = template_mask_2d.astype(dtype)

        template_dgram = dgram_from_positions(batch['template_pseudo_beta'],
                                              **self.config.dgram_features)
        template_dgram = template_dgram.astype(dtype)

        aatype = nn.functional.one_hot(batch['template_aatype'], 22)
        aatype = aatype.astype(dtype)

        to_concat = [template_dgram, template_mask_2d[..., None]]
        to_concat.append(
            paddle.tile(aatype[..., None, :, :], [1, num_res, 1, 1]))
        to_concat.append(paddle.tile(aatype[..., None, :], [1, 1, num_res, 1]))

        n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]
        rot, trans = quat_affine.make_transform_from_reference(
            n_xyz=batch['template_all_atom_positions'][..., n, :],
            ca_xyz=batch['template_all_atom_positions'][..., ca, :],
            c_xyz=batch['template_all_atom_positions'][..., c, :])
        affines = quat_affine.QuatAffine(
            quaternion=quat_affine.rot_to_quat(rot),
            translation=trans,
            rotation=rot)

        points = [
            paddle.unsqueeze(
                x, axis=-2) for x in paddle.unstack(
                    affines.translation, axis=-1)
        ]
        affine_vec = affines.invert_point(points, extra_dims=1)
        inv_distance_scalar = paddle.rsqrt(1e-6 + sum(
            [paddle.square(x) for x in affine_vec]))

        # Backbone affine mask: whether the residue has C, CA, N
        # (the template mask defined above only considers pseudo CB).
        template_mask = (batch['template_all_atom_masks'][..., n] *
                         batch['template_all_atom_masks'][..., ca] *
                         batch['template_all_atom_masks'][..., c])
        template_mask_2d = template_mask[..., None] * template_mask[...,
                                                                    None, :]
        inv_distance_scalar *= template_mask_2d.astype(
            inv_distance_scalar.dtype)

        unit_vector = [(x * inv_distance_scalar)[..., None]
                       for x in affine_vec]
        unit_vector = [x.astype(dtype) for x in unit_vector]
        if not self.config.use_template_unit_vector:
            unit_vector = [paddle.zeros_like(x) for x in unit_vector]
        to_concat.extend(unit_vector)

        template_mask_2d = template_mask_2d.astype(dtype)
        to_concat.append(template_mask_2d[..., None])

        act = paddle.concat(to_concat, axis=-1)
        # Mask out non-template regions so we don't get arbitrary values in the
        # distogram for these regions.
        act *= template_mask_2d[..., None]

        act = self.embedding2d(act)

        act = dap.scatter(act, axis=1)
        for idx, pair_encoder in enumerate(self.template_pair_stack):
            act = recompute_wrapper(
                pair_encoder,
                act,
                mask_2d,
                is_recompute=self.training and idx >=
                self.config.template_pair_stack.recompute_start_block_index)
        act = dap.gather(act, axis=1)

        act = self.output_layer_norm(act)
        return act


class TemplateEmbedding(nn.Layer):
    """Embeds a set of templates.

        Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-12
        Jumper et al. (2021) Suppl. Alg. 17 "TemplatePointwiseAttention"
    """

    def __init__(self, channel_num, config, global_config):
        super(TemplateEmbedding, self).__init__()
        self.config = config
        self.global_config = global_config

        self.single_template_embedding = SingleTemplateEmbedding(
            channel_num, config, global_config)
        self.attention = Attention(
            config.attention, global_config, channel_num['pair_channel'],
            config.attention.key_dim, channel_num['pair_channel'])

    def forward(self, query_embedding, template_batch, mask_2d):
        """Build TemplateEmbedding module.

        Arguments:
            query_embedding: Query pair representation, shape [n_batch, N_res, N_res, c_z].
            template_batch: A batch of template features.
            mask_2d: Padding mask (Note: this doesn't care if a template exists,
                unlike the template_pseudo_beta_mask).

        Returns:
            A template embedding [n_batch, N_res, N_res, c_z].
        """

        num_templates = template_batch['template_mask'].shape[1]

        num_channels = (self.config.template_pair_stack.
                        triangle_attention_ending_node.value_dim)

        num_res = query_embedding.shape[1]

        dtype = query_embedding.dtype
        template_mask = template_batch['template_mask']
        template_mask = template_mask.astype(dtype)

        query_channels = query_embedding.shape[-1]

        outs = []
        for i in range(num_templates):
            # By default, num_templates = 4
            batch0 = {
                k: paddle.squeeze(
                    v.slice([1], [i], [i + 1]), axis=1)
                for k, v in template_batch.items()
            }
            outs.append(
                self.single_template_embedding(query_embedding, batch0,
                                               mask_2d))

        template_pair_repr = paddle.stack(outs, axis=1)

        flat_query = paddle.reshape(
            query_embedding, [-1, num_res * num_res, 1, query_channels])
        flat_templates = paddle.reshape(
            paddle.transpose(template_pair_repr, [0, 2, 3, 1, 4]),
            [-1, num_res * num_res, num_templates, num_channels])

        bias = 1e9 * (template_mask[:, None, None, None, :] - 1.)

        if not self.training:
            sb_attn = subbatch(self.attention, [0, 1], [1, 1],
                               self.config.subbatch_size, 1)
            emb = sb_attn(flat_query, flat_templates, bias)

        else:
            emb = self.attention(flat_query, flat_templates, bias)

        emb = paddle.reshape(emb, [-1, num_res, num_res, query_channels])

        # No gradients if no templates.
        emb *= (paddle.sum(template_mask) > 0.).astype(emb.dtype)
        return emb
