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

import numpy as np
import paddle
import paddle.nn as nn

try:
    from paddle import _legacy_C_ops as _C_ops
except:
    from paddle import _C_ops

from ppfleetx.distributed.protein_folding import dap

from .common import (
    init_gate_linear,
    init_final_linear,
    mask_mean,
    subbatch, )


class Attention(nn.Layer):
    """Multihead attention."""

    def __init__(self, config, global_config, q_dim, kv_dim, output_dim):
        super(Attention, self).__init__()
        self.config = config
        self.global_config = global_config

        num_head = self.config.num_head
        key_dim = self.config.get('key_dim', q_dim)
        value_dim = self.config.get('value_dim', kv_dim)

        # TODO(GuoxiaWang): delete non fuse_attention related code on dcu
        self.fuse_attention = self.global_config.fuse_attention
        self.merge_qkv = (q_dim == kv_dim)

        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_dim = key_dim
        self.value_dim = value_dim

        self.qkv_w = None
        self.query_w = None
        self.key_w = None
        self.value_w = None
        if self.merge_qkv and self.fuse_attention:
            self.qkv_w = paddle.create_parameter(
                [3, num_head, key_dim, q_dim],
                'float32',
                default_initializer=nn.initializer.XavierUniform())
        else:
            self.query_w = paddle.create_parameter(
                [q_dim, num_head, key_dim],
                'float32',
                default_initializer=nn.initializer.XavierUniform())
            self.key_w = paddle.create_parameter(
                [kv_dim, num_head, key_dim],
                'float32',
                default_initializer=nn.initializer.XavierUniform())
            self.value_w = paddle.create_parameter(
                [kv_dim, num_head, value_dim],
                'float32',
                default_initializer=nn.initializer.XavierUniform())

        self.gating_w = None
        self.gating_b = None
        if self.config.gating:
            self.gating_w = paddle.create_parameter(
                [q_dim, num_head, value_dim],
                'float32',
                default_initializer=nn.initializer.Constant(0.0))
            self.gating_b = paddle.create_parameter(
                [num_head, value_dim],
                'float32',
                default_initializer=nn.initializer.Constant(1.0))

        if self.global_config.zero_init:
            init = nn.initializer.Constant(0.0)
        else:
            init = nn.initializer.XavierUniform()

        self.output_w = paddle.create_parameter(
            [num_head, value_dim, output_dim],
            'float32',
            default_initializer=init)
        self.output_b = paddle.create_parameter(
            [output_dim],
            'float32',
            default_initializer=nn.initializer.Constant(0.0))

    def forward(self, q_data, m_data, bias, nonbatched_bias=None):
        """Builds Attention module.
        Arguments:
            q_data: A tensor of queries, shape [batch, row_size, N_queries, q_channels].
            m_data: A tensor of memories from which the keys and values are
                projected, shape [batch, row_size, N_keys, m_channels].
            bias: A bias for the attention, shape [batch, row_size, num_head, N_queries, N_keys].
            nonbatched_bias: Shared bias, shape [N_queries, N_keys].

        Returns:
            A float32 tensor of shape [batch_size, row_size, N_queries, output_dim].
        """
        if self.fuse_attention:
            if nonbatched_bias is not None:
                nonbatched_bias = paddle.unsqueeze(nonbatched_bias, axis=1)
            _, _, _, _, _, _, _, output = _C_ops.fused_gate_attention(
                q_data, m_data, self.query_w, self.key_w, self.value_w,
                self.qkv_w, nonbatched_bias, bias, self.gating_w,
                self.gating_b, self.output_w, self.output_b, 'has_gating',
                self.config.gating, 'merge_qkv', self.merge_qkv)
        else:
            c = self.key_dim**(-0.5)
            q = paddle.einsum('nbqa,ahc->nbqhc', q_data, self.query_w) * c
            k = paddle.einsum('nbka,ahc->nbkhc', m_data, self.key_w)
            v = paddle.einsum('nbka,ahc->nbkhc', m_data, self.value_w)
            logits = paddle.einsum('nbqhc,nbkhc->nbhqk', q, k) + bias

            if nonbatched_bias is not None:
                logits += paddle.unsqueeze(nonbatched_bias, axis=1)

            weights = nn.functional.softmax(logits)
            weighted_avg = paddle.einsum('nbhqk,nbkhc->nbqhc', weights, v)

            if self.config.gating:
                gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                            self.gating_w) + self.gating_b
                gate_values = nn.functional.sigmoid(gate_values)
                weighted_avg *= gate_values

            output = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                                   self.output_w) + self.output_b
        return output


class GlobalAttention(nn.Layer):
    """Global attention.

    Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention" lines 2-7
    """

    def __init__(self, config, global_config, q_dim, kv_dim, output_dim):
        super(GlobalAttention, self).__init__()
        self.config = config
        self.global_config = global_config

        num_head = self.config.num_head
        key_dim = self.config.get('key_dim', q_dim)
        value_dim = self.config.get('value_dim', kv_dim)

        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query_w = paddle.create_parameter(
            [q_dim, num_head, key_dim],
            'float32',
            default_initializer=nn.initializer.XavierUniform())
        self.key_w = paddle.create_parameter(
            [kv_dim, key_dim],
            'float32',
            default_initializer=nn.initializer.XavierUniform())
        self.value_w = paddle.create_parameter(
            [kv_dim, value_dim],
            'float32',
            default_initializer=nn.initializer.XavierUniform())

        if self.config.gating:
            self.gating_w = paddle.create_parameter(
                [q_dim, num_head, value_dim],
                'float32',
                default_initializer=nn.initializer.Constant(0.0))
            self.gating_b = paddle.create_parameter(
                [num_head, value_dim],
                'float32',
                default_initializer=nn.initializer.Constant(1.0))

        if self.global_config.zero_init:
            init = nn.initializer.Constant(0.0)
        else:
            init = nn.initializer.XavierUniform()

        self.output_w = paddle.create_parameter(
            [num_head, value_dim, output_dim],
            'float32',
            default_initializer=init)
        self.output_b = paddle.create_parameter(
            [output_dim],
            'float32',
            default_initializer=nn.initializer.Constant(0.0))

    def forward(self, q_data, m_data, q_mask):
        k = paddle.einsum('nbka,ac->nbkc', m_data, self.key_w)
        v = paddle.einsum('nbka,ac->nbkc', m_data, self.value_w)

        # NOTE: differ from non-global version using q_avg for attn
        q_avg = mask_mean(q_mask, q_data, axis=2)
        c = self.key_dim**(-0.5)
        q = paddle.einsum('nba,ahc->nbhc', q_avg, self.query_w) * c

        q_mask_ = paddle.unsqueeze(q_mask, axis=2)[..., 0]
        bias = 1e9 * (q_mask_ - 1.)

        logits = paddle.einsum('nbhc,nbkc->nbhk', q, k) + bias
        weights = nn.functional.softmax(logits)
        weighted_avg = paddle.einsum('nbhk,nbkc->nbhc', weights, v)

        if self.config.gating:
            gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                        self.gating_w) + self.gating_b
            gate_values = nn.functional.sigmoid(gate_values)
            weighted_avg = paddle.unsqueeze(weighted_avg, axis=2)
            weighted_avg *= gate_values

            output = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                                   self.output_w) + self.output_b
        else:
            output = paddle.einsum('nbhc,hco->nbo', weighted_avg,
                                   self.output_w) + self.output_b
            output = paddle.unsqueeze(output, axis=-1)

        return output


class MSARowAttentionWithPairBias(nn.Layer):
    """MSA per-row attention biased by the pair representation.

    Jumper et al. (2021) Suppl. Alg. 7 "MSARowAttentionWithPairBias"
    """

    def __init__(self, channel_num, config, global_config, is_extra_msa):
        super(MSARowAttentionWithPairBias, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.is_extra_msa = is_extra_msa
        assert config.orientation == 'per_row'

        if is_extra_msa:
            self.query_norm = nn.LayerNorm(channel_num['extra_msa_channel'])
        else:
            self.query_norm = nn.LayerNorm(channel_num['msa_channel'])

        self.feat_2d_norm = nn.LayerNorm(channel_num['pair_channel'])
        self.feat_2d_weights = paddle.create_parameter(
            [channel_num['pair_channel'], self.config.num_head],
            'float32',
            default_initializer=nn.initializer.Normal(
                std=1. / np.sqrt(channel_num['pair_channel'])))

        if is_extra_msa:
            extra_msa_channel = channel_num['extra_msa_channel']
            self.attention = Attention(self.config, self.global_config,
                                       extra_msa_channel, extra_msa_channel,
                                       extra_msa_channel)
        else:
            msa_channel = channel_num['msa_channel']
            self.attention = Attention(self.config, self.global_config,
                                       msa_channel, msa_channel, msa_channel)

    def forward(self, msa_act, msa_mask, pair_act):

        pair_act = self.feat_2d_norm(pair_act)

        # [B, N_res//dap_size, N_res, cz], [cz, head] => [B, head, N_res//dap_size, N_res]
        nonbatched_bias_before = paddle.einsum('nqkc,ch->nhqk', pair_act,
                                               self.feat_2d_weights)

        # [B, head, N_res//dap_size, N_res] => [B, head, N_res, N_res]
        nonbatched_bias = dap.all_gather(nonbatched_bias_before, axis=2)
        nonbatched_bias = dap.all_gather_opp(nonbatched_bias, axis=2)

        # [B, N_seq, N_res] => [B, N_seq//dap_size, N_res]
        msa_mask = dap.scatter(msa_mask, axis=1)

        bias = 1e9 * (msa_mask - 1.)
        # [B, N_seq//dap_size, N_res] => [B, N_seq//dap_size, 1, 1, N_res]
        bias = paddle.unsqueeze(bias, axis=[2, 3])
        msa_act = self.query_norm(msa_act)

        if not self.training or (self.is_extra_msa and
                                 self.config.use_subbatch):
            # low memory mode using subbatch
            subbatch_size = self.config.subbatch_size
            if not self.training:
                subbatch_size = self.global_config.subbatch_size
            sb_attn = subbatch(
                self.attention, [0, 1, 2], [1, 1, 1],
                subbatch_size,
                1,
                same_arg_idx={1: 0})
            msa_act = sb_attn(msa_act, msa_act, bias, nonbatched_bias)
        else:
            msa_act = self.attention(msa_act, msa_act, bias, nonbatched_bias)

        return msa_act


class MSAColumnGlobalAttention(nn.Layer):
    """MSA per-column global attention.

    Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention"
    """

    def __init__(self, channel_num, config, global_config):
        super(MSAColumnGlobalAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        assert config.orientation == 'per_column'

        extra_msa_channel = channel_num['extra_msa_channel']
        self.query_norm = nn.LayerNorm(extra_msa_channel)
        self.attention = GlobalAttention(self.config, self.global_config,
                                         extra_msa_channel, extra_msa_channel,
                                         extra_msa_channel)

    def forward(self, msa_act, msa_mask):
        # scatter if using dap, otherwise do nothing
        # [B, N_seq, N_res] => [B, N_seq, N_res//dap_size]
        msa_mask = dap.scatter(msa_mask, axis=2)

        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        msa_mask = paddle.transpose(msa_mask, [0, 2, 1])

        bias = 1e9 * (msa_mask - 1.)
        bias = paddle.unsqueeze(bias, axis=[2, 3])

        msa_mask = paddle.unsqueeze(msa_mask, axis=-1)
        msa_act = self.query_norm(msa_act)

        if not self.training:
            # low memory mode using subbatch
            sb_attn = subbatch(
                self.attention, [0, 1, 2], [1, 1, 1],
                self.global_config.subbatch_size,
                1,
                same_arg_idx={1: 0})
            msa_act = sb_attn(msa_act, msa_act, msa_mask)
        else:
            msa_act = self.attention(msa_act, msa_act, msa_mask)

        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        return msa_act


class MSAColumnAttention(nn.Layer):
    """MSA per-column attention.

    Jumper et al. (2021) Suppl. Alg. 8 "MSAColumnAttention"
    """

    def __init__(self, channel_num, config, global_config):
        super(MSAColumnAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        assert config.orientation == 'per_column'

        msa_channel = channel_num['msa_channel']
        self.query_norm = nn.LayerNorm(msa_channel)
        self.attention = Attention(self.config, self.global_config,
                                   msa_channel, msa_channel, msa_channel)

    def forward(self, msa_act, msa_mask):
        # scatter if using dap, otherwise do nothing
        # [B, N_seq, N_res] => [B, N_seq, N_res//dap_size]
        msa_mask = dap.scatter(msa_mask, axis=2)

        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        msa_mask = paddle.transpose(msa_mask, [0, 2, 1])

        bias = 1e9 * (msa_mask - 1.)
        bias = paddle.unsqueeze(bias, axis=[2, 3])

        msa_act = self.query_norm(msa_act)
        if not self.training:
            # low memory mode using subbatch
            sb_attn = subbatch(
                self.attention, [0, 1, 2], [1, 1, 1],
                self.global_config.subbatch_size,
                1,
                same_arg_idx={1: 0})
            msa_act = sb_attn(msa_act, msa_act, bias)
        else:
            msa_act = self.attention(msa_act, msa_act, bias)

        msa_act = paddle.transpose(msa_act, [0, 2, 1, 3])
        return msa_act


class TriangleAttention(nn.Layer):
    """Triangle Attention.

    Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
    Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
    """

    def __init__(self,
                 channel_num,
                 config,
                 global_config,
                 name='triangle_attention'):
        super(TriangleAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        assert config.orientation in ['per_row', 'per_column']

        self.query_norm = nn.LayerNorm(
            channel_num['pair_channel'], name='query_norm')
        self.feat_2d_weights = paddle.create_parameter(
            [channel_num['pair_channel'], self.config.num_head],
            'float32',
            default_initializer=nn.initializer.Normal(
                std=1. / np.sqrt(channel_num['pair_channel'])))

        self.attention = Attention(
            self.config, self.global_config, channel_num['pair_channel'],
            channel_num['pair_channel'], channel_num['pair_channel'])

    def forward(self, pair_act, pair_mask):
        """Builds TriangleAttention module.

        Arguments:
        pair_act: [batch, N_res, N_res, c_z] pair activations tensor
        pair_mask: [batch, N_res, N_res] mask of non-padded regions in the tensor.

        Returns:
        Update to pair_act, shape [batch, N_res, N_res, c_z].
        """
        if self.config.orientation == 'per_column':
            pair_act = pair_act.transpose([0, 2, 1, 3])
            pair_mask = pair_mask.transpose([0, 2, 1])

        # [B, N_res//dap_size, N_res]
        bias = 1e9 * (pair_mask - 1.)
        # [B, N_res//dap_size, 1, 1, N_res]
        bias = paddle.unsqueeze(bias, axis=[2, 3])

        pair_act = self.query_norm(pair_act)

        # [B, N_res//dap_size, N_res, cz], [cz, head] => [B, head, N_res//dap_size, N_res]
        nonbatched_bias_before = paddle.einsum('bqkc,ch->bhqk', pair_act,
                                               self.feat_2d_weights)

        # # [B, head, N_res//dap_size, N_res] => [B, head, N_res, N_res]
        nonbatched_bias = dap.all_gather(nonbatched_bias_before, axis=2)
        nonbatched_bias = dap.all_gather_opp(nonbatched_bias, axis=2)

        if not self.training:
            # low memory mode using subbatch
            sb_attn = subbatch(
                self.attention, [0, 1, 2], [1, 1, 1],
                self.global_config.subbatch_size,
                1,
                same_arg_idx={1: 0})
            pair_act = sb_attn(pair_act, pair_act, bias, nonbatched_bias)
        else:
            pair_act = self.attention(pair_act, pair_act, bias,
                                      nonbatched_bias)

        if self.config.orientation == 'per_column':
            pair_act = pair_act.transpose([0, 2, 1, 3])

        return pair_act


class TriangleMultiplication(nn.Layer):
    """Triangle multiplication layer ("outgoing" or "incoming").

    Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
    Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"
    """

    def __init__(self,
                 channel_num,
                 config,
                 global_config,
                 name='triangle_multiplication'):
        super(TriangleMultiplication, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.layer_norm_input = nn.LayerNorm(
            self.channel_num['pair_channel'], name='layer_norm_input')
        self.left_projection = nn.Linear(
            self.channel_num['pair_channel'],
            self.config.num_intermediate_channel,
            name='left_projection')
        self.right_projection = nn.Linear(
            self.channel_num['pair_channel'],
            self.config.num_intermediate_channel,
            name='right_projection')
        self.left_gate = nn.Linear(
            self.channel_num['pair_channel'],
            self.config.num_intermediate_channel,
            name='left_gate')
        init_gate_linear(self.left_gate)
        self.right_gate = nn.Linear(
            self.channel_num['pair_channel'],
            self.config.num_intermediate_channel,
            name='right_gate')
        init_gate_linear(self.right_gate)

        # line 4
        self.center_layer_norm = nn.LayerNorm(
            self.config.num_intermediate_channel, name='center_layer_norm')
        self.output_projection = nn.Linear(
            self.config.num_intermediate_channel,
            self.channel_num['pair_channel'],
            name='output_projection')
        init_final_linear(self.output_projection)
        # line 3
        self.gating_linear = nn.Linear(
            self.channel_num['pair_channel'],
            self.channel_num['pair_channel'],
            name='output_projection')
        init_gate_linear(self.gating_linear)

    def forward(self, act, mask):
        """Builds TriangleMultiplication module.

        Arguments:
        act: Pair activations, shape [batch, N_res, N_res, c_z]
        mask: Pair mask, shape [batch, N_res, N_res].

        Returns:
        Outputs, same shape/type as act.
        """
        # Outgoing [batch, N_res//dap_size, N_res] => [batch, N_res//dap_size, N_res, 1]
        # Incoming [batch, N_res, N_res//dap_size] => [batch, N_res, N_res//dap_size, 1] 
        mask = paddle.unsqueeze(mask, axis=-1)  # [batch, N_res, N_res, 1]

        # Outgoing [B, N_res//dap_size, N_res, c_z]
        # Incoming [B, N_res, N_res//dap_size, c_z]
        act = self.layer_norm_input(act)  # line 1

        # Outgoing [B, N_res//dap_size, N_res, c_z] => [B, N_res//dap_size, N_res, num_intermediate_channel]
        # Incoming [B, N_res, N_res//dap_size, c_z] => [B, N_res, N_res//dap_size, num_intermediate_channel]
        left_proj_act = mask * self.left_projection(act)
        right_proj_act = mask * self.right_projection(act)

        # Outgoing [B, N_res//dap_size, N_res, c_z] => [B, N_res//dap_size, N_res, num_intermediate_channel]
        # Incoming [B, N_res, N_res//dap_size, c_z] => [B, N_res, N_res//dap_size, num_intermediate_channel]
        left_gate_values = nn.functional.sigmoid(self.left_gate(act))
        right_gate_values = nn.functional.sigmoid(self.right_gate(act))

        # Outgoing [B, N_res//dap_size, N_res, num_intermediate_channel]
        # Incoming [B, N_res, N_res//dap_size, num_intermediate_channel]
        left_proj_act = left_proj_act * left_gate_values
        right_proj_act_before = right_proj_act * right_gate_values

        # "Outgoing" edges equation: 'ikc,jkc->ijc'
        # "Incoming" edges equation: 'kjc,kic->ijc'
        # Note on the Suppl. Alg. 11 & 12 notation:
        # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
        # For the "incoming" edges, it's swapped:
        #   b = left_proj_act and a = right_proj_act

        if self.config.equation == 'ikc,jkc->ijc':
            # Outgoing
            # [B, N_res//dap_size, N_res, num_intermediate_channel] => [B, N_res, N_res, num_intermediate_channel]
            right_proj_act = dap.all_gather(right_proj_act_before, axis=1)
        elif self.config.equation == 'kjc,kic->ijc':
            # Incoming
            # [B, N_res, N_res//dap_size, num_intermediate_channel] => [B, N_res, N_res, num_intermediate_channel]
            right_proj_act = dap.all_gather(right_proj_act_before, axis=2)
        else:
            raise ValueError('unknown equation.')

        # Outgoing [B, N_res//dap_size, N_res, c_z]
        # Incoming [B, N_res, N_res//dap_size, c_z]        
        gate_values = nn.functional.sigmoid(self.gating_linear(act))  # line 3

        if self.config.equation == 'ikc,jkc->ijc':
            # Outgoing
            dim, out_idx = 1, 1
            equation = 'bikc,bjkc->bijc'

            # [B, N_res, N_res, num_intermediate_channel]
            right_proj_act_after = dap.all_gather_opp(right_proj_act, axis=1)
        elif self.config.equation == 'kjc,kic->ijc':
            # Incoming
            dim, out_idx = 2, 2
            equation = 'bkjc,bkic->bijc'

            # [B, N_res, N_res, num_intermediate_channel]
            right_proj_act_after = dap.all_gather_opp(right_proj_act, axis=2)
        else:
            raise ValueError('unknown equation.')

        if not self.training:
            einsum_fn = subbatch(paddle.einsum, [1], [dim],
                                 self.global_config.subbatch_size, out_idx)
            act = einsum_fn(equation, left_proj_act, right_proj_act_after)
        else:
            # Outgoing equation = 'bikc,bjkc->bijc'
            # [B, N_res//dap_size, N_res, num_intermediate_channel], [B, N_res, N_res, num_intermediate_channel]
            # => [B, N_res//dap_size, N_res, num_intermediate_channel]

            # Incoming equation = 'bkjc,bkic->bijc'
            # [B, N_res, N_res//dap_size, num_intermediate_channel], [B, N_res, N_res, num_intermediate_channel]
            # => [B, N_res, N_res//dap_size, num_intermediate_channel]
            act = paddle.einsum(equation, left_proj_act, right_proj_act_after)

        act = self.center_layer_norm(act)
        act = self.output_projection(act)

        act = act * gate_values

        return act
