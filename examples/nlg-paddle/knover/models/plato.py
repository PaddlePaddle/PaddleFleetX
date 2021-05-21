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
"""Plato model."""

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from knover.models import register_model
from knover.core.model import Model
from knover.models.unified_transformer import UnifiedTransformer
from knover.modules.transformer_block import pre_process_layer
from knover.utils import repeat_array_or_tensor
from knover.utils import str2bool


@register_model("Plato")
class Plato(UnifiedTransformer):
    """Plato model."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = UnifiedTransformer.add_cmdline_args(parser)
        group.add_argument("--use_bow", type=str2bool, default=True,
                           help="Whether to use BoW loss in training.")
        group.add_argument("--use_entropy", type=str2bool, default=False,
                           help="Whether to use entropy loss in training.")
        group.add_argument("--entropy_bound", type=float, default=None)
        group.add_argument("--entropy_loss_coef", type=float, default=0.01)
        group.add_argument("--use_nsp", type=str2bool, default=False)
        return group

    def __init__(self, args, place):
        # latent related
        self.mask_id = args.mask_id
        self.latent_type_size = args.latent_type_size
        self.latent_emb_name = "latent_embedding"
        self.use_bow = args.use_bow
        self.use_entropy = args.use_entropy
        self.entropy_bound = args.entropy_bound
        self.entropy_loss_coef = args.entropy_loss_coef
        self.use_nsp = args.use_nsp

        super(Plato, self).__init__(args, place)

    def _get_feed_dict(self, is_infer=False):
        """Get model's input feed dict.

        Args:
            is_infer: If true, get inference input feed dict, otherwise get training / evaluation input feed dict.

        Returns:
            feed_dict: A feed dict mapping keys to feed input variable.
        """
        feed_dict = {}
        feed_dict["token_ids"] = paddle.static.data(name="token_ids", shape=[-1, -1, 1], dtype="int64")
        feed_dict["type_ids"] = paddle.static.data(name="type_ids", shape=[-1, -1, 1], dtype="int64")
        feed_dict["pos_ids"] = paddle.static.data(name="pos_ids", shape=[-1, -1, 1], dtype="int64")
        if self.use_role:
            feed_dict["role_ids"] = paddle.static.data(name="role_ids", shape=[-1, -1, 1], dtype="int64")
        if self.use_turn:
            feed_dict["turn_ids"] = paddle.static.data(name="turn_ids", shape=[-1, -1, 1], dtype="int64")

        if self.use_nsp:
            feed_dict["neg_token_ids"] = paddle.static.data(
                name="neg_token_ids", shape=[-1, -1, 1], dtype="int64")
            feed_dict["neg_type_ids"] = paddle.static.data(
                name="neg_type_ids", shape=[-1, -1, 1], dtype="int64")
            feed_dict["neg_pos_ids"] = paddle.static.data(
                name="neg_pos_ids", shape=[-1, -1, 1], dtype="int64")
            if self.use_role:
                feed_dict["neg_role_ids"] = paddle.static.data(
                    name="neg_role_ids", shape=[-1, -1, 1], dtype="int64")
            if self.use_turn:
                feed_dict["neg_turn_ids"] = paddle.static.data(
                    name="neg_turn_ids", shape=[-1, -1, 1], dtype="int64")

        if not is_infer:
            feed_dict["recognition_mask"] = paddle.static.data(
                name="recognition_mask",
                shape=[-1, -1, -1],
                dtype=self.dtype)
            if self.use_nsp:
                feed_dict["neg_recognition_mask"] = paddle.static.data(
                    name="neg_recognition_mask",
                    shape=[-1, -1, -1],
                    dtype=self.dtype)
        feed_dict["generation_mask"] = paddle.static.data(
            name="generation_mask",
            shape=[-1, -1, -1],
            dtype=self.dtype)

        if is_infer:
            feed_dict["tgt_ids"] = paddle.static.data(
                name="tgt_ids", shape=[-1, -1, 1], dtype="int64", lod_level=2)
            feed_dict["tgt_pos"] = paddle.static.data(
                name="tgt_pos", shape=[-1, -1, 1], dtype="int64", lod_level=2)
            feed_dict["init_score"] = paddle.static.data(
                name="init_score", shape=[-1, 1], dtype="float32", lod_level=1)
            feed_dict["parent_idx"] = paddle.static.data(
                name="parent_idx", shape=[-1], dtype="int64")

            feed_dict["tgt_generation_mask"] = paddle.static.data(
                name="tgt_generation_mask", shape=[-1, 1, -1], dtype="float32")
            feed_dict["latent_id"] = paddle.static.data(name="latent_id", shape=[-1, 1], dtype="int64")
            feed_dict["data_id"] = paddle.static.data(name="data_id", shape=[-1, 1], dtype="int64")
        else:
            feed_dict["tgt_label"] = paddle.static.data(name="tgt_label", shape=[-1, 1], dtype="int64")
            feed_dict["tgt_idx"] = paddle.static.data(name="tgt_idx", shape=[-1, 2], dtype="int64")

            if self.use_bow:
                feed_dict["bow_label"] = paddle.static.data(name="bow_label", shape=[-1, 1], dtype="int64")
                feed_dict["bow_idx"] = paddle.static.data(name="bow_idx", shape=[-1, 2], dtype="int64")

        return feed_dict

    def _recognition_network(self,
                             token_ids,
                             type_ids,
                             pos_ids,
                             role_ids,
                             turn_ids,
                             input_mask):
        """Run recognition network.

        Args:
            tokens_ids: represents the token id of each token, shape is [batch_size, max_seq_len, 1]
            type_ids: represents the type of each token, shape is [batch_size, max_seq_len, 1]
            pos_ids: represents the position of each token, shape is [batch_size, max_seq_len, 1]
            input_mask: represents the attention masking mastrix in each Transformer blocks,
                shape is [batch_size, max_seq_len + 1, max_seq_len + 1]

        Returns:
            A tuple contains the output embeddings of Transformer and the checkpoints of Transformer in this pass.
        """
        mask_id = layers.fill_constant_batch_size_like(
            input=token_ids, shape=[-1, 1, 1], value=self.mask_id, dtype="int64")
        mask_emb = paddle.static.nn.embedding(
            input=mask_id,
            size=[self.vocab_size, self.emb_size],
            dtype=self.dtype,
            param_attr=paddle.ParamAttr(
                name=self.token_emb_name, initializer=self.param_initializer))
        emb_out, attn_bias = self._gen_input(
            token_ids, type_ids, pos_ids, role_ids, turn_ids, input_mask, aux_emb=mask_emb)

        return self._encode(emb_out, attn_bias)

    def _calc_recognition_logits(self, enc_out):
        """Get the logits of latent recognition task.

        The network may share weight with latent embeddings.
        Args:
            enc_out: the output embeddings of Transformer, shape is [batch_size, max_seq_len, hidden_dim]

        Returns:
            logits: the logits of prediction task, shape is [batch_size, latent_type_size].
        """
        recognition_feat = self._get_pooled_output(enc_out, name="recognition")
        logits = paddle.static.nn.fc(
            x=recognition_feat,
            size=self.latent_type_size,
            weight_attr=paddle.ParamAttr(name=self.latent_emb_name, initializer=self.param_initializer),
            bias_attr="recognition_bias")
        return logits

    def _calc_nsp_logits(self, recognition_out):
        recognition_feat = self._get_pooled_output(recognition_out, name="nsp")
        nsp_logits = self._get_classifier_output(recognition_feat, name="next_sent")
        return nsp_logits

    def forward(self, inputs, is_infer=False):
        """Run model main forward."""
        outputs = {}
        if is_infer:
            self.generation_caches = [{
                "k":
                layers.fill_constant_batch_size_like(
                    input=inputs["token_ids"],
                    shape=[-1, 0, self.d_key * self.n_head],
                    dtype=self.dtype,
                    value=0),
                "v":
                layers.fill_constant_batch_size_like(
                    input=inputs["token_ids"],
                    shape=[-1, 0, self.d_value * self.n_head],
                    dtype=self.dtype,
                    value=0),
            } for i in range(self.n_layer)]
        else:
            self.generation_caches = None

        latent_embeddings = paddle.create_parameter(
            shape=[self.emb_size, self.latent_type_size],
            dtype=self.dtype,
            attr=paddle.ParamAttr(
                name=self.latent_emb_name, initializer=self.param_initializer))

        if is_infer:
            latent_id = inputs["latent_id"]
            weights = layers.one_hot(latent_id, self.latent_type_size)
        else:
            recognition_outputs = self._recognition_network(
                token_ids=inputs["token_ids"],
                type_ids=inputs["type_ids"],
                pos_ids=inputs["pos_ids"],
                role_ids=inputs.get("role_ids", None),
                turn_ids=inputs.get("turn_ids", None),
                input_mask=inputs["recognition_mask"],
            )
            logits = self._calc_recognition_logits(recognition_outputs["enc_out"])

            outputs["post_probs"] = paddle.nn.functional.softmax(logits)
            weights = gumbel_softmax(logits)
            outputs["checkpoints"] = recognition_outputs["checkpoints"]

            if self.use_nsp:
                outputs["pos_logits"] = self._calc_nsp_logits(recognition_outputs["enc_out"])
                neg_recognition_outputs = self._recognition_network(
                    token_ids=inputs["neg_token_ids"],
                    type_ids=inputs["neg_type_ids"],
                    pos_ids=inputs["neg_pos_ids"],
                    role_ids=inputs.get("neg_role_ids", None),
                    turn_ids=inputs.get("neg_turn_ids", None),
                    input_mask=inputs["neg_recognition_mask"])
                outputs["neg_logits"] = self._calc_nsp_logits(neg_recognition_outputs["enc_out"])
                outputs["checkpoints"].extend(neg_recognition_outputs["checkpoints"])

        latent_emb = paddle.matmul(x=weights, y=latent_embeddings, transpose_y=True)
        generation_outputs = self._generation_network(
            token_ids=inputs["token_ids"],
            type_ids=inputs["type_ids"],
            pos_ids=inputs["pos_ids"],
            role_ids=inputs.get("role_ids", None),
            turn_ids=inputs.get("turn_ids", None),
            generation_mask=inputs["generation_mask"],
            aux_emb=paddle.unsqueeze(latent_emb, axis=[1]),
            gather_idx=inputs.get("parent_idx", None),
        )
        outputs["enc_out"] = generation_outputs["enc_out"]

        if not is_infer:
            outputs["checkpoints"].extend(generation_outputs["checkpoints"])
        return outputs

    def _calc_bow_logits(self, enc_out, bow_idx):
        """Get the logits of BoW task.

        The network may share weight with token embeddings.

        Args:
            enc_out: the output embeddings of Transformer, shape is [batch_size, max_seq_len, hidden_dim]
            bow_idx: the indices of prediction tokens, shape is [num_predictions, 1] or [num_predictions, 2].

        Returns:
            logits: the logits of prediction task, shape is [num_predictions, vocab_size].
        """
        if len(bow_idx.shape) == 2 and bow_idx.shape[1] == 1:
            enc_out = paddle.squeeze(x=enc_out, axis=[1])
            bow_feat = paddle.gather(x=enc_out, index=bow_idx)
        elif len(bow_idx.shape) == 2 and bow_idx.shape[1] == 2:
            bow_feat = paddle.gather_nd(x=enc_out, index=bow_idx)
            # bow_feat = layers.gather(input=enc_out[:, 0], index=bow_idx[:, :1], overwrite=False)
        else:
            raise ValueError(f"Invalid indices shape {bow_idx.shape} is used")

        bow_trans_feat = paddle.static.nn.fc(
            x=bow_feat,
            size=self.emb_size,
            activation=self.hidden_act,
            weight_attr=paddle.ParamAttr(
                name="bow_trans_fc.w_0",
                initializer=self.param_initializer),
            bias_attr="bow_trans_fc.b_0")

        bow_trans_feat = pre_process_layer(
            bow_trans_feat, self.post_cls_cmd, name="bow_trans")

        if self.weight_sharing:
            bow_logits = paddle.matmul(
                x=bow_trans_feat,
                y=paddle.static.default_main_program().global_block().var(
                    self.token_emb_name),
                transpose_y=True)
            if self.cls_bias:
                bow_logits += paddle.create_parameter(
                    shape=[self.vocab_size],
                    dtype=self.dtype,
                    attr=paddle.ParamAttr(name="bow_out_fc.b_0"),
                    is_bias=True)
        else:
            bow_out_bias_attr = "bow_out_fc.b_0" if self.cls_bias else False
            bow_logits = paddle.static.nn.fc(input=bow_trans_feat,
                                   size=self.vocab_size,
                                   weight_attr=paddle.ParamAttr(
                                       name="bow_out_fc.w_0",
                                       initializer=self.param_initializer),
                                   bias_attr=bow_out_bias_attr)
        return bow_logits

    def get_metrics(self, inputs, outputs):
        """Get metrics."""
        metrics = super(Plato, self).get_metrics(inputs, outputs)

        if self.use_bow:
            bow_logits = self._calc_bow_logits(outputs["enc_out"], inputs["bow_idx"])
            bow_loss = paddle.nn.functional.softmax_with_cross_entropy(
                logits=bow_logits, label=inputs["bow_label"])
            mean_bow_loss = paddle.mean(bow_loss)
            metrics["token_bow_loss"] = mean_bow_loss
            metrics["loss"] = metrics["loss"] + mean_bow_loss

        entropy_loss = paddle.sum(outputs["post_probs"] * paddle.log(outputs["post_probs"]), axis=1)
        mean_entropy_loss = paddle.mean(entropy_loss)
        metrics["entropy_loss"] = mean_entropy_loss
        if self.use_entropy:
            if self.entropy_bound:
                cond = entropy_loss > paddle.ones_like(x=entropy_loss) * self.entropy_bound
                cond = paddle.cast(cond, dtype="float32")
                entropy_loss = cond * (entropy_loss - self.entropy_bound)
                mean_entropy_loss = paddle.mean(x=entropy_loss)
            metrics["loss"] = metrics["loss"] + self.entropy_loss_coef * mean_entropy_loss

        if self.use_nsp:
            logits = paddle.concat([outputs["pos_logits"], outputs["neg_logits"]], axis=0)
            label = paddle.concat(
                [
                    layers.fill_constant_batch_size_like(outputs["pos_logits"], [-1, 1], "int64", 1),
                    layers.fill_constant_batch_size_like(outputs["neg_logits"], [-1, 1], "int64", 0),
                ],
                axis=0
            )
            nsp_loss, nsp_softmax = paddle.nn.functional.softmax_with_cross_entropy(
                logits=logits, label=label, return_softmax=True)
            nsp_acc = paddle.metric.accuracy(nsp_softmax, label)
            mean_nsp_acc = paddle.mean(x=nsp_acc)
            mean_nsp_loss = paddle.mean(x=nsp_loss)
            metrics["nsp_loss"] = mean_nsp_loss
            metrics["nsp_acc"] = mean_nsp_acc
            metrics["loss"] = metrics["loss"] + mean_nsp_loss
        return metrics

    def infer_step(self, inputs):
        """Run one inference step."""
        # handle DataLoader input type in distributed mode.
        if isinstance(inputs, list):
            inputs = inputs[0]
        if self.do_generation:
            batch_size = self._get_batch_size(inputs)
            inputs = {
                name: repeat_array_or_tensor(array_or_tensor, self.place, self.latent_type_size)
                for name, array_or_tensor in inputs.items()
            }
            # Add latent_id
            inputs["latent_id"] = np.array(
                [i for i in range(self.latent_type_size) for _ in range(batch_size)],
                dtype="int64"
            ).reshape([-1, 1])

            return super(Plato, self).infer_step(inputs)
        else:
            return self._execute(
                self.infer_program,
                inputs,
                self.infer_fetch_dict)


def gumbel_softmax(logits, tau=0.67, eps=1e-10):
    """Gumbel softmax."""
    u = layers.uniform_random_batch_size_like(
        logits, shape=[-1, logits.shape[1]], min=0.0, max=1.0)
    u.stop_gradient = True
    gumbel = 0.0 - paddle.log(eps - paddle.log(u + eps))
    y = logits + gumbel
    return paddle.nn.functional.softmax(x=y / tau)
