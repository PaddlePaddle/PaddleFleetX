# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import math
import copy
import json
import numpy as np
from collections import OrderedDict
from typing import Callable, List, Optional, Set, Tuple, Union, Any

import paddle
from paddle import nn

from ppfleetx.data.tokenizers.t5_tokenizer import (
    t5_tokenize, get_t5_tokenizer, DEFAULT_T5_NAME)
from ppfleetx.models.multimodal_model.imagen.utils import rearrange, exists, default


def finfo(dtype):
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


def fields(class_or_instance):
    """Return a tuple describing the fields of this dataclass.

    Accepts a dataclass or an instance of one. Tuple elements are of
    type Field.
    """

    # Might it be worth caching this, per class?
    try:
        fields = getattr(class_or_instance, _FIELDS)
    except AttributeError:
        raise TypeError('must be called with a dataclass type or instance')

    # Exclude pseudo-fields.  Note that fields is sorted by insertion
    # order, so the order of the tuple is as the fields were defined.
    return tuple(f for f in fields.values() if f._field_type is _FIELD)


def is_tensor(x):
    return isinstance(x, paddle.Tensor)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f"{self.__class__.__name__} should not have more than one required field."
            )

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (not isinstance(element, (list, tuple)) or
                            not len(element) == 2 or
                            not isinstance(element[0], str)):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


class NewGELUActivation(nn.Layer):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + paddle.tanh(
            math.sqrt(2.0 / math.pi) *
            (input + 0.044715 * paddle.pow(input, 3.0))))


class GELUActivation(nn.Layer):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    paddle.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * paddle.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool=False):
        super().__init__()
        self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + paddle.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)


class FastGELUActivation(nn.Layer):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input):
        return 0.5 * input * (
            1.0 + paddle.tanh(input * 0.7978845608 *
                              (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Layer):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input):
        return input * paddle.nn.functional.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Layer):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    paddle.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * paddle.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(
                f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return paddle.clip(gelu(x), self.min, self.max)


class SiLUActivation(nn.Layer):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.functional.silu

    def _silu_python(self, input):
        return input * nn.functional.sigmoid(input)

    def forward(self, input):
        return self.act(input)


class MishActivation(nn.Layer):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        super().__init__()
        self.act = nn.functional.mish

    def _mish_python(self, input):
        return input * paddle.tanh(nn.functional.softplus(input))

    def forward(self, input):
        return self.act(input)


class LinearActivation(nn.Layer):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input):
        return input


ACT2FN = {
    "gelu": GELUActivation(),
    "gelu_10": ClippedGELUActivation(-10, 10),
    "gelu_fast": FastGELUActivation(),
    "gelu_new": NewGELUActivation(),
    "gelu_python": GELUActivation(use_gelu_python=True),
    "linear": LinearActivation(),
    "mish": MishActivation(),
    "quick_gelu": QuickGELUActivation(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": SiLUActivation(),
    "swish": SiLUActivation(),
    "tanh": nn.Tanh(),
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}"
        )


# For backwards compatibility with: from activations import gelu_python
gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")


def prune_linear_layer(layer: nn.Linear, index: paddle.int64,
                       dim: int=0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`paddle.nn.Linear`): The layer to prune.
        index (`paddle.int64`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `paddle.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(
        new_size[1], new_size[0], bias_attr=layer.bias is not None)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W)
    new_layer.weight.stop_gradient = False
    if layer.bias is not None:
        new_layer.bias.stop_gradient = True
        new_layer.bias.copy_(b)
        new_layer.bias.stop_gradient = False
    return new_layer


def find_pruneable_heads_and_indices(heads,
                                     n_heads: int,
                                     head_size: int,
                                     already_pruned_heads):
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads : List of the indices of heads to prune.
        n_heads : The number of heads in the model.
        head_size : The size of each head.
        already_pruned_heads : A set of already pruned heads.

    Returns:
        A tuple with the remaining heads and their corresponding indices.
    """
    mask = paddle.ones(n_heads, head_size)
    heads = set(
        heads
    ) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.reshape(-1).equal(1)
    index = paddle.arange(len(mask))[mask].cast(paddle.int64)
    return heads, index


class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state = None
    past_key_values = None
    hidden_states = None
    attentions = None
    cross_attentions = None


class T5Config(object):
    def __init__(self, **kwargs):

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.use_return_dict = kwargs.pop("return_dict", True)
        self.d_ff = kwargs.pop("d_ff", None)
        self.d_kv = kwargs.pop("d_kv", None)
        self.d_model = kwargs.pop("d_model", None)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id",
                                                 None)
        self.dense_act_fn = kwargs.pop("dense_act_fn", 'gelu_new')
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.feed_forward_proj = kwargs.pop("feed_forward_proj", None)
        self.initializer_factor = kwargs.pop("initializer_factor", None)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_gated_act = kwargs.pop("is_gated_act", True)
        self.layer_norm_epsilon = kwargs.pop("layer_norm_epsilon", None)
        self.model_type = kwargs.pop("model_type", None)
        self.num_decoder_layers = kwargs.pop("num_decoder_layers", None)
        self.num_heads = kwargs.pop("num_heads", None)
        self.num_layers = kwargs.pop("num_layers", None)
        self.output_past = kwargs.pop("output_past", True)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.relative_attention_max_distance = kwargs.pop(
            "relative_attention_max_distance", 128)
        self.relative_attention_num_buckets = kwargs.pop(
            "relative_attention_num_buckets", None)
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        self.transformers_version = kwargs.pop("transformers_version", None)
        self.use_cache = kwargs.pop("use_cache", False)
        self.vocab_size = kwargs.pop("vocab_size", None)
        self.model_type = kwargs.pop("model_type", None)
        self.dropout_rate = kwargs.pop("dropout_rate", None)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)


class T5LayerNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = self.create_parameter(
            [hidden_size],
            default_initializer=nn.initializer.Constant(value=1.))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.cast(paddle.float32).pow(2).mean(
            -1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance +
                                                     self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = hidden_states.cast(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseActDense(nn.Layer):
    def __init__(self, d_model, d_ff, dropout_rate, dense_act_fn):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias_attr=False)
        self.wo = nn.Linear(d_ff, d_model, bias_attr=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = ACT2FN[dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Layer):
    def __init__(self, d_model, d_ff, dropout_rate, dense_act_fn):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias_attr=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias_attr=False)
        self.wo = nn.Linear(d_ff, d_model, bias_attr=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = ACT2FN[dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Layer):
    def __init__(self, d_model, d_ff, dropout_rate, layer_norm_epsilon,
                 feed_forward_proj):
        super().__init__()
        if feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedActDense(
                d_model, d_ff, dropout_rate, dense_act_fn)
        elif feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseActDense(d_model, d_ff, dropout_rate,
                                                  feed_forward_proj)

        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Layer):
    def __init__(self,
                 is_decoder,
                 relative_attention_num_buckets,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.d_model = d_model
        self.key_value_proj_dim = d_kv
        self.n_heads = num_heads
        self.dropout = dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias_attr=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads)
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  bidirectional=True,
                                  num_buckets=32,
                                  max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                relative_position > 0).cast(paddle.int64) * num_buckets
            relative_position = paddle.abs(relative_position)
        else:
            relative_position = -paddle.min(
                relative_position, paddle.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            paddle.log(relative_position.cast('float32') /
                       max_exact) / math.log(max_distance / max_exact) *
            (num_buckets - max_exact)).cast(paddle.int64)
        relative_position_if_large = paddle.minimum(
            relative_position_if_large,
            paddle.full_like(relative_position_if_large, num_buckets - 1))

        relative_buckets += paddle.where(is_small, relative_position,
                                         relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        context_position = paddle.arange(
            query_length, dtype=paddle.int64)[:, None]
        memory_position = paddle.arange(
            key_length, dtype=paddle.int64)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets, )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.transpose([2, 0, 1]).unsqueeze(
            0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
            self,
            hidden_states,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False, ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[
                2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[
            1]

        def shape(states):
            """projection"""
            return states.reshape(
                [0, -1, self.n_heads, self.key_value_proj_dim]).transpose(
                    [0, 2, 1, 3])

        def unshape(states):
            """reshape"""
            return states.transpose([0, 2, 1, 3]).reshape(
                [batch_size, -1, self.inner_dim])

        def project(hidden_states, proj_layer, key_value_states,
                    past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = paddle.concat(
                        [past_key_value, hidden_states], axis=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(
            hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(hidden_states, self.k, key_value_states,
                             past_key_value[0]
                             if past_key_value is not None else None)
        value_states = project(hidden_states, self.v, key_value_states,
                               past_key_value[1]
                               if past_key_value is not None else None)

        # compute scores
        scores = paddle.matmul(
            query_states, key_states.transpose([0, 1, 3, 2])
        )  # equivalent of paddle.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = paddle.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    dtype=scores.dtype)
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(
            scores.cast('float32'), axis=-1).astype(
                scores.dtype)  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.
            training)  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(paddle.matmul(
            attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (
            self.is_decoder and use_cache) else None
        outputs = (attn_output, ) + (present_key_value_state, ) + (
            position_bias, )

        if output_attentions:
            outputs = outputs + (attn_weights, )
        return outputs


class T5LayerSelfAttention(nn.Layer):
    def __init__(self,
                 is_decoder,
                 relative_attention_num_buckets,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 layer_norm_epsilon,
                 has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(
            is_decoder,
            relative_attention_num_buckets,
            d_model,
            d_kv,
            num_heads,
            dropout_rate,
            has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False, ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions, )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,
                   ) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Layer):
    def __init__(self, is_decoder, relative_attention_num_buckets, d_model,
                 d_kv, num_heads, dropout_rate, layer_norm_epsilon):
        super().__init__()
        self.EncDecAttention = T5Attention(
            is_decoder,
            relative_attention_num_buckets,
            d_model,
            d_kv,
            num_heads,
            has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
            self,
            hidden_states,
            key_value_states,
            attention_mask=None,
            position_bias=None,
            layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            query_length=None,
            output_attentions=False, ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions, )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,
                   ) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Layer):
    def __init__(self,
                 is_decoder,
                 relative_attention_num_buckets,
                 feed_forward_proj,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 layer_norm_epsilon,
                 d_ff,
                 has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.LayerList()
        self.layer.append(
            T5LayerSelfAttention(
                is_decoder,
                relative_attention_num_buckets,
                d_model,
                d_kv,
                num_heads,
                dropout_rate,
                layer_norm_epsilon,
                has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(
                T5LayerCrossAttention(
                    is_decoder, relative_attention_num_buckets, d_model, d_kv,
                    num_heads, dropout_rate, layer_norm_epsilon))

        self.layer.append(
            T5LayerFF(d_model, d_ff, dropout_rate, layer_norm_epsilon,
                      feed_forward_proj))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            layer_head_mask=None,
            cross_attn_layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            return_dict=True, ):

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning(
                    "`past_key_values` is passed to the encoder. Please make sure this is intended."
                )
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states")

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions, )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[
            2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == paddle.float16 and paddle.isinf(
                hidden_states).any():
            clamp_value = finfo(hidden_states.dtype).max - 1000
            hidden_states = paddle.clip(
                hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions, )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == paddle.float16 and paddle.isinf(
                    hidden_states).any():
                clamp_value = finfo(hidden_states.dtype).max - 1000
                hidden_states = paddle.clip(
                    hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[
                    1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == paddle.float16 and paddle.isinf(
                hidden_states).any():
            clamp_value = finfo(hidden_states.dtype).max - 1000
            hidden_states = paddle.clip(
                hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states, )

        if use_cache:
            outputs = outputs + (present_key_value_state, ) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5Stack(nn.Layer):
    def __init__(self,
                 d_model,
                 num_layers,
                 layer_norm_epsilon,
                 dropout_rate,
                 relative_attention_num_buckets,
                 feed_forward_proj,
                 d_kv,
                 num_heads,
                 d_ff,
                 embed_tokens=None,
                 is_decoder=False):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.is_decoder = is_decoder
        self.num_layers = num_layers

        self.block = nn.LayerList([
            T5Block(
                is_decoder,
                relative_attention_num_buckets,
                feed_forward_proj,
                d_model,
                d_kv,
                num_heads,
                dropout_rate,
                layer_norm_epsilon,
                d_ff,
                has_relative_attention_bias=bool(i == 0))
            for i in range(num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def get_extended_attention_mask(self, attention_mask, input_shape):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`paddle.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `paddle.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = extended_attention_mask.cast(dtype='float16')  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self,
                      head_mask,
                      num_hidden_layers,
                      is_attention_chunked=False):
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`paddle.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `paddle.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask,
                                                      num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                -1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                -1)  # We can specify head_mask for each layer
        assert head_mask.dim(
        ) == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        #head_mask = head_mask.cast(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True, ):
        if use_cache is True:
            assert (
                self.is_decoder
            ), f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else False)

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape([-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[
            2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = paddle.ones(batch_size, mask_seq_length)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = paddle.ones(
                batch_size, encoder_seq_length, dtype=paddle.int64)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size,
                                    encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask,
                                                  self.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and
                                      self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value
                ) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions, )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,
                                                     ) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state, )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3], )
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (
                        layer_outputs[5], )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v
                         for v in [
                             hidden_states,
                             present_key_value_states,
                             all_hidden_states,
                             all_attentions,
                             all_cross_attentions,
                         ] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions, )


class T5EncoderModel(nn.Layer):
    authorized_missing_keys = [r"encoder.embed_tokens.weight", ]

    def __init__(self,
                 vocab_size=32128,
                 d_model=768,
                 d_kv=64,
                 d_ff=3072,
                 num_layers=12,
                 num_decoder_layers=12,
                 num_heads=12,
                 relative_attention_num_buckets=32,
                 dropout_rate=0.1,
                 layer_norm_epsilon=1e-06,
                 feed_forward_proj="relu"):
        super().__init__()
        self.shared = nn.Embedding(vocab_size, d_model)
        # self.extra_parameters = list(self.shared.parameters())

        use_cache = False
        is_encoder_decoder = False
        self.encoder = T5Stack(
            d_model,
            num_layers,
            layer_norm_epsilon,
            dropout_rate,
            relative_attention_num_buckets,
            feed_forward_proj,
            d_kv,
            num_heads,
            d_ff,
            embed_tokens=self.shared,
            is_decoder=False)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import T5Tokenizer, T5EncoderModel

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5EncoderModel.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else True
        #import numpy as np
        #attention_mask = paddle.to_tensor(np.load('attn_mask.npy'))
        #input_ids = paddle.to_tensor(np.load('input_ids.npy'))
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        return encoder_outputs


def T5Model(config):
    config = T5Config(**config)
    model = T5EncoderModel(config)
    return model


def get_t5_model(name, pretrained=True):
    #t5_config = dict_from_json_file(name)
    #model = T5Model(t5_config)
    model = T5EncoderModel(
        vocab_size=32128,
        d_model=1024,
        d_kv=128,
        d_ff=65536,
        num_layers=2,
        num_decoder_layers=None,
        num_heads=128,
        relative_attention_num_buckets=32,
        dropout_rate=0.,
        layer_norm_epsilon=1e-06,
        feed_forward_proj="relu")
    if pretrained:
        checkpoint = paddle.load(name + '/t5.pd', return_numpy=True)
        model.set_state_dict(checkpoint['model'])
    model.eval()
    for p in model.parameters():
        p.stop_gradient = True

    return model


def t5_11b():
    return T5EncoderModel(
        vocab_size=32128,
        d_model=1024,
        d_kv=128,
        d_ff=65536,
        num_layers=24,
        num_decoder_layers=None,
        num_heads=128,
        relative_attention_num_buckets=32,
        dropout_rate=0.,
        layer_norm_epsilon=1e-06,
        feed_forward_proj="relu")


def dict_from_json_file(name):
    with open(name + '/config.json', "r", encoding="utf-8") as reader:
        text = reader.read()
        config_dict = json.loads(text)
        return config_dict


def t5_encode_text(t5, texts, tokenizer, return_attn_mask=False):
    token_ids, attn_mask = t5_tokenize(texts, tokenizer)
    t5.eval()
    with paddle.no_grad():
        encoded_text = t5(input_ids=token_ids, attention_mask=attn_mask)
        text_features = encoded_text.last_hidden_state.detach()

    if return_attn_mask:
        #attn_mask = attn_mask.cast('bool')
        return text_features, attn_mask

    return text_features


def get_encoded_dim(name):
    return dict_from_json_file(name)['d_model']
