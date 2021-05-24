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
"""Unified Transformer model."""

import numpy as np
import paddle

from knover.models import register_model
from knover.core.model import Model
from knover.models.unified_transformer import UnifiedTransformer
from knover.modules.generator import Generator
from knover.modules.moe_transformer_block import moe_encoder
from knover.utils import str2bool, repeat_array_or_tensor, slice_array_or_tensor


@register_model("SparseMoE")
class SparseMoE(UnifiedTransformer):
    """Unified Transformer"""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = UnifiedTransformer.add_cmdline_args(parser)
        group.add_argument("--aux_loss_coef", type=float, default=0.0)
        return group

    def __init__(self, args, place):
        self.num_experts = args.num_experts
        self.experts_capacity = args.experts_capacity
        self.aux_loss_coef = args.aux_loss_coef

        super(SparseMoE, self).__init__(args, place)
        return

    def _encode(self,
                emb_input,
                attn_bias,
                caches=None,
                gather_idx=None,
                name="encoder"):
        """Run Transformer encode pass.

        Args:
            emb_input: represents the input embeddings of Transformer, shape is [batch_size, max_seq_len, hidden_size]
            attn_bias: represents the attention masking matrix, shape is [batch_size, 1, max_seq_len, max_seq_len]
            caches: a dict, the caches used in efficient decoding, which cache Ks and Vs of memory in each MHA.
            gather_idx: a index tensor, which determine which branch is used to generate next token.

        Returns:
            A tuple contains the output embeddings of Transformer and the checkpoints of Transformer in this pass.
        """
        padding_mask = paddle.cast(
            paddle.max(attn_bias[:, 0], 2, keepdim=True) < 0,
            "float32")
        enc_out, checkpoints, aux_loss = moe_encoder(
            enc_input=emb_input,
            attn_bias=attn_bias,
            n_layer=self.n_layer,
            n_head=self.n_head,
            d_key=self.d_key,
            d_value=self.d_value,
            d_model=self.hidden_size,
            d_inner_hid=self.inner_hidden_size,
            num_experts=self.num_experts,
            experts_capacity=self.experts_capacity,
            prepostprocess_dropout=self.prepostprocess_dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=0,
            hidden_act=self.hidden_act,
            padding_mask=padding_mask,
            pre_encoder_cmd=self.pre_encoder_cmd,
            preprocess_cmd=self.preprocess_cmd,
            postprocess_cmd=self.postprocess_cmd,
            param_initializer=self.param_initializer,
            epsilon=self.epsilon,
            n_layer_per_block=self.n_layer_per_block,
            param_share=self.param_share,
            name=name,
            caches=caches,
            gather_idx=gather_idx,
            store=caches is not None
        )
        outputs = {
            "enc_out": enc_out,
            "checkpoints": checkpoints
        }
        if aux_loss is not None:
            outputs["aux_loss"] = aux_loss * self.aux_loss_coef
        return outputs
