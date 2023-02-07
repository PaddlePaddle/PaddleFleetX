# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization for DebertaV2."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import json
import copy
import logging
import warnings
import regex as re
import unicodedata
import sentencepiece as sp
from collections import OrderedDict, UserDict
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from ppfleetx.utils.download import cached_path
from ppfleetx.data.tokenizers.tokenization_utils_base import (
    _LazyConfigMapping, AddedToken, TruncationStrategy, PaddingStrategy,
    BatchEncoding, SpecialTokensMixin)

logger = logging.getLogger(__name__)

MAX_LENGTH = 256

DEFAULT_DebertaV2_NAME = "projects/imagen/cache/deberta-v-xxlarge"

# Slow tokenizers used to be saved in three separated files
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
FULL_TOKENIZER_FILE = "tokenizer.json"
_re_tokenizer_file = re.compile(r"tokenizer\.(.*)\.json")

CONFIG_NAME = "config.json"


def get_debertav2_tokenizer(name):
    tokenizer = DebertaV2Tokenizer.from_pretrained(name)
    return tokenizer


def debertav2_tokenize(texts, tokenizer):
    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors="paddle",
        padding='longest',
        max_length=MAX_LENGTH,
        truncation=True)

    input_ids = encoded.input_ids
    attn_mask = encoded.attention_mask
    return input_ids, attn_mask


PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-v2-xlarge":
        "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xxlarge":
        "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xlarge-mnli":
        ("https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/spm.model"
         ),
        "microsoft/deberta-v2-xxlarge-mnli":
        ("https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/spm.model"
         ),
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-v2-xlarge": 512,
    "microsoft/deberta-v2-xxlarge": 512,
    "microsoft/deberta-v2-xlarge-mnli": 512,
    "microsoft/deberta-v2-xxlarge-mnli": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-v2-xlarge": {
        "do_lower_case": False
    },
    "microsoft/deberta-v2-xxlarge": {
        "do_lower_case": False
    },
    "microsoft/deberta-v2-xlarge-mnli": {
        "do_lower_case": False
    },
    "microsoft/deberta-v2-xxlarge-mnli": {
        "do_lower_case": False
    },
}

VOCAB_FILES_NAMES = {"vocab_file": "spm.model"}


class DebertaV2Tokenizer(SpecialTokensMixin):
    r"""
    Constructs a DeBERTa-v2 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        bos_token (`string`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.
            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.
        eos_token (`string`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token. When building a sequence using special tokens, this is not the token that is
            used for the end of sequence. The token used is the `sep_token`.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    padding_side = "right"
    truncation_side = "right"
    slow_tokenizer_class = None

    def __init__(self,
                 vocab_file,
                 do_lower_case=False,
                 split_by_punct=False,
                 bos_token="[CLS]",
                 eos_token="[SEP]",
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 sp_model_kwargs=None,
                 **kwargs):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}

        super().__init__(
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            split_by_punct=split_by_punct,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs, )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.do_lower_case = do_lower_case
        self.split_by_punct = split_by_punct
        self.vocab_file = vocab_file
        self._tokenizer = SPMTokenizer(
            vocab_file,
            split_by_punct=split_by_punct,
            sp_model_kwargs=self.sp_model_kwargs)

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return self.vocab_size + len(self.added_tokens_encoder)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs,
                        **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)
        _raise_exceptions_for_missing_entries = False

        user_agent = {
            "file_type": "tokenizer",
            "from_auto_class": from_auto_class,
            "is_fast": "Fast" in cls.__name__
        }
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        vocab_files = {}
        init_configuration = {}

        is_local = os.path.isdir(pretrained_model_name_or_path)
        single_file_id = None
        if os.path.isfile(
                pretrained_model_name_or_path
        ):  # or is_remote_url(pretrained_model_name_or_path):
            if len(cls.vocab_files_names) > 1:
                raise ValueError(
                    f"Calling {cls.__name__}.from_pretrained() with the path to a single file or url is not "
                    "supported for this tokenizer. Use a model identifier or the path to a directory instead."
                )
            warnings.warn(
                f"Calling {cls.__name__}.from_pretrained() with the path to a single file or url is deprecated and "
                "won't be possible anymore in v5. Use a model identifier or the path to a directory instead.",
                FutureWarning, )
            file_id = list(cls.vocab_files_names.keys())[0]

            vocab_files[file_id] = pretrained_model_name_or_path
            single_file_id = file_id
        else:
            # At this point pretrained_model_name_or_path is either a directory or a model identifier name
            additional_files_names = {
                "added_tokens_file": ADDED_TOKENS_FILE,
                "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,
                "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
            }
            vocab_files = {
                ** cls.vocab_files_names, ** additional_files_names
            }

            if "tokenizer_file" in vocab_files:
                # Try to get the tokenizer config to see if there are versioned tokenizer files.
                fast_tokenizer_file = FULL_TOKENIZER_FILE
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    TOKENIZER_CONFIG_FILE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    local_files_only=local_files_only,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    _commit_hash=commit_hash, )
                commit_hash = extract_commit_hash(resolved_config_file,
                                                  commit_hash)
                if resolved_config_file is not None:
                    with open(
                            resolved_config_file, encoding="utf-8") as reader:
                        tokenizer_config = json.load(reader)
                        if "fast_tokenizer_files" in tokenizer_config:
                            fast_tokenizer_file = get_fast_tokenizer_file(
                                tokenizer_config["fast_tokenizer_files"])
                vocab_files["tokenizer_file"] = fast_tokenizer_file

        # Get files from url, cache, or disk depending on the case
        resolved_vocab_files = {}
        unresolved_files = []
        for file_id, file_path in vocab_files.items():
            if file_path is None:
                resolved_vocab_files[file_id] = None
            elif single_file_id == file_id:
                if os.path.isfile(file_path):
                    resolved_vocab_files[file_id] = file_path
                elif is_remote_url(file_path):
                    resolved_vocab_files[file_id] = download_url(
                        file_path, proxies=proxies)
            else:
                if subfolder is None:
                    subfolder = ""
                path_or_repo_id = str(pretrained_model_name_or_path)
                if os.path.isdir(path_or_repo_id):
                    resolved_file = os.path.join(
                        os.path.join(path_or_repo_id, subfolder), file_path)
                    if not os.path.isfile(resolved_file):
                        if _raise_exceptions_for_missing_entries:
                            raise EnvironmentError(
                                f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                                f"'https://huggingface.co/{path_or_repo_id}/{revision}' for available files."
                            )
                        else:
                            resolved_file = None
                    resolved_vocab_files[file_id] = resolved_file

                else:
                    resolved_vocab_files[file_id] = cached_path(
                        file_path,
                        cache_dir=cache_dir, )

        if len(unresolved_files) > 0:
            logger.info(
                f"Can't load following files from cache: {unresolved_files} and cannot check if these "
                "files are necessary for the tokenizer to operate.")

        if all(full_file_name is None
               for full_file_name in resolved_vocab_files.values()):
            raise EnvironmentError(
                f"Can't load tokenizer for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing all relevant files for a {cls.__name__} tokenizer."
            )

        for file_id, file_path in vocab_files.items():
            if file_id not in resolved_vocab_files:
                continue

            #if is_local:
            #    logger.info(f"loading file {file_path}")
            #else:
            #    logger.info(f"loading file {file_path} from cache at {resolved_vocab_files[file_id]}")

        return cls._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            _commit_hash=commit_hash,
            **kwargs, )

    @classmethod
    def _from_pretrained(cls,
                         resolved_vocab_files,
                         pretrained_model_name_or_path,
                         init_configuration,
                         *init_inputs,
                         use_auth_token=None,
                         cache_dir=None,
                         local_files_only=False,
                         _commit_hash=None,
                         **kwargs):
        # We instantiate fast tokenizers based on a slow tokenizer if we don't have access to the tokenizer.json
        # file or if `from_slow` is set to True.
        from_slow = kwargs.get("from_slow", False)
        has_tokenizer_file = resolved_vocab_files.get("tokenizer_file",
                                                      None) is not None
        if from_slow:
            slow_tokenizer = (cls.slow_tokenizer_class)._from_pretrained(
                copy.deepcopy(resolved_vocab_files),
                pretrained_model_name_or_path,
                copy.deepcopy(init_configuration),
                *init_inputs,
                use_auth_token=use_auth_token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                _commit_hash=_commit_hash,
                **(copy.deepcopy(kwargs)), )
        else:
            slow_tokenizer = None

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop(
            "tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with open(
                    tokenizer_config_file,
                    encoding="utf-8") as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            # First attempt. We get tokenizer_class from tokenizer_config to check mismatch between tokenizers.
            config_tokenizer_class = init_kwargs.get("tokenizer_class")
            init_kwargs.pop("tokenizer_class", None)
            init_kwargs.pop("auto_map", None)
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            config_tokenizer_class = None
            init_kwargs = init_configuration

        if config_tokenizer_class is None:
            try:
                config_dict = resolved_vocab_files.pop("config_file",
                                                       CONFIG_NAME)
                config_dict = os.path.join(pretrained_model_name_or_path,
                                           config_dict)
                config_dict = cls._dict_from_json_file(config_dict)
                config_tokenizer_class = config_dict[
                    "tokenizer_class"] if "tokenizer_class" in config_dict else None
            except (OSError, ValueError, KeyError):
                # skip if an error occurred.
                config_dict = None
            if config_tokenizer_class is None:
                # Third attempt. If we have not yet found the original type of the tokenizer,
                # we are loading we see if we can infer it from the type of the configuration file
                from ppfleetx.data.tokenizers.tokenization_utils_base import TOKENIZER_MAPPING_NAMES  # tests_ignore

                model_type = config_dict[
                    "model_type"] if "model_type" in config_dict else None
                if model_type is None:
                    # Fallback: use pattern matching on the string.
                    for pattern in TOKENIZER_MAPPING_NAMES.keys():
                        if pattern in str(pretrained_model_name_or_path):
                            model_type = pattern
                            break

                if model_type is not None:
                    config_tokenizer_class, config_tokenizer_class_fast = TOKENIZER_MAPPING_NAMES.get(
                        model_type, (None, None))
                    if config_tokenizer_class is None:
                        config_tokenizer_class = config_tokenizer_class_fast

        if config_tokenizer_class is not None:
            if cls.__name__.replace(
                    "Fast", "") != config_tokenizer_class.replace("Fast", ""):
                logger.warning(
                    "The tokenizer class you load from this checkpoint is not the same type as the class this"
                    " function is called from. It may result in unexpected tokenization. \nThe tokenizer class you"
                    f" load from this checkpoint is '{config_tokenizer_class}'. \nThe class this function is called"
                    f" from is '{cls.__name__}'.")

        # Update with newly provided kwargs
        init_kwargs.update(kwargs)

        # Convert AddedTokens serialized as dict to class instances
        def convert_added_tokens(obj: Union[AddedToken, Any]):
            if isinstance(obj, dict) and "__type" in obj and obj[
                    "__type"] == "AddedToken":
                obj.pop("__type")
                return AddedToken(**obj)
            elif isinstance(obj, (list, tuple)):
                return list(convert_added_tokens(o) for o in obj)
            elif isinstance(obj, dict):
                return {k: convert_added_tokens(v) for k, v in obj.items()}
            return obj

        init_kwargs = convert_added_tokens(init_kwargs)

        # Set max length if needed
        if pretrained_model_name_or_path in cls.max_model_input_sizes:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings

            model_max_length = cls.max_model_input_sizes[
                pretrained_model_name_or_path]
            if model_max_length is not None and isinstance(model_max_length,
                                                           (int, float)):

                model_max_length = min(
                    init_kwargs.get("model_max_length", int(1e30)),
                    model_max_length)
                # TODO(PVP) - uncomment following line in Transformers v5
                # init_kwargs["model_max_length"] = model_max_length
                # TODO(PVP) - remove in Transformers v5
                # ---
                init_kwargs[
                    "model_max_length"] = cls._eventually_correct_t5_max_length(
                        pretrained_model_name_or_path, model_max_length,
                        init_kwargs.get("model_max_length"))
                # ---

            # Merge resolved_vocab_files arguments in init_kwargs.
        added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path

        if slow_tokenizer is not None:
            init_kwargs["__slow_tokenizer"] = slow_tokenizer

        init_kwargs["name_or_path"] = pretrained_model_name_or_path

        # Instantiate tokenizer.
        try:
            tokenizer = cls(*init_inputs, **init_kwargs)
        except OSError:
            raise OSError(
                "Unable to load vocabulary from file. "
                "Please check that the provided vocabulary is accessible and not corrupted."
            )

        # Save inputs and kwargs for saving and re-loading with ``save_pretrained``
        # Removed: Now done at the base class level
        # tokenizer.init_inputs = init_inputs
        # tokenizer.init_kwargs = init_kwargs

        # If there is a complementary special token map, load it
        special_tokens_map_file = resolved_vocab_files.pop(
            "special_tokens_map_file", None)
        if special_tokens_map_file is not None:
            with open(
                    special_tokens_map_file,
                    encoding="utf-8") as special_tokens_map_handle:
                special_tokens_map = json.load(special_tokens_map_handle)
            for key, value in special_tokens_map.items():
                if key in kwargs and kwargs[key]:
                    # This value has already been redefined by the kwargs
                    # We keep this new value and ignore the one stored in the special_tokens_map_file

                    continue

                if isinstance(value, dict):
                    value = AddedToken(**value)
                elif isinstance(value, list):
                    value = [
                        AddedToken(**token)
                        if isinstance(token, dict) else token
                        for token in value
                    ]
                setattr(tokenizer, key, value)

        # Add supplementary tokens.
        special_tokens = tokenizer.all_special_tokens
        if added_tokens_file is not None:
            with open(
                    added_tokens_file,
                    encoding="utf-8") as added_tokens_handle:
                added_tok_encoder = json.load(added_tokens_handle)

            # Sort added tokens by index
            added_tok_encoder_sorted = list(
                sorted(
                    added_tok_encoder.items(), key=lambda x: x[1]))

            # Accumulate added tokens into batches of special/non-special tokens, because calling add_tokens() for
            # individual tokens would repeatedly rebuild a trie, which can be slow.
            is_last_special = None
            tokens = []

            for token, index in added_tok_encoder_sorted:
                current_index = len(tokenizer) + len(tokens)
                if has_tokenizer_file and index != current_index and tokenizer.convert_tokens_to_ids(
                        token) != index:
                    # Tokenizer fast: added token needs to either be in the vocabulary with the proper index or the
                    # index is the current length of the tokenizer (not in vocabulary)
                    raise ValueError(
                        f"Wrong index found for {token}: should be {tokenizer.convert_tokens_to_ids(token)} but found "
                        f"{index}.")
                elif not has_tokenizer_file and index != current_index:
                    # Tokenizer slow: added token cannot already be in the vocabulary so its index needs to be the
                    # current length of the tokenizer.
                    raise ValueError(
                        f"Non-consecutive added token '{token}' found. "
                        f"Should have index {current_index} but has index {index} in saved vocabulary."
                    )

                is_special = bool(token in special_tokens)
                if is_last_special is None or is_last_special == is_special:
                    tokens.append(token)
                else:
                    tokenizer.add_tokens(
                        tokens, special_tokens=is_last_special)
                    tokens = [token]
                is_last_special = is_special

            if tokens:
                tokenizer.add_tokens(tokens, special_tokens=is_last_special)

        # Check all our special tokens are registered as "no split" token (we don't cut them) and are in the vocab
        added_tokens = tokenizer.sanitize_special_tokens()
        #if added_tokens:
        #    logger.warning_advice(
        #        "Special tokens have been added in the vocabulary, make sure the associated word embeddings are"
        #        " fine-tuned or trained."
        #    )

        return tokenizer

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab(self):
        return self._tokenizer.vocab

    def get_vocab(self):
        vocab = self.vocab.copy()
        vocab.update(self.get_added_vocab())
        return vocab

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        if self.do_lower_case:
            text = text.lower()
        return self._tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self._tokenizer.spm.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self._tokenizer.spm.IdToPiece(
            index) if index < self.vocab_size else self.unk_token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        return self._tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True)

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 +
                                                        sep) * [1]

    def prepare_for_tokenization(self,
                                 text,
                                 is_split_into_words=False,
                                 **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def save_vocabulary(self,
                        save_directory: str,
                        filename_prefix: Optional[str]=None) -> Tuple[str]:
        return self._tokenizer.save_pretrained(
            save_directory, filename_prefix=filename_prefix)

    def _eventual_warn_about_too_long_sequence(self,
                                               ids,
                                               max_length,
                                               verbose: bool):
        """
        Depending on the input and internal state we might trigger a warning about a sequence that is too long for its
        corresponding model

        Args:
            ids (`List[str]`): The ids produced by the tokenization
            max_length (`int`, *optional*): The max_length desired (does not trigger a warning if it is set)
            verbose (`bool`): Whether or not to print more information and warnings.

        """
        if max_length is None and len(ids) > self.model_max_length and verbose:
            if not self.deprecation_warnings.get(
                    "sequence-length-is-longer-than-the-specified-maximum",
                    False):
                logger.warning(
                    "Token indices sequence length is longer than the specified maximum sequence length "
                    f"for this model ({len(ids)} > {self.model_max_length}). Running this sequence through the model "
                    "will result in indexing errors")
            self.deprecation_warnings[
                "sequence-length-is-longer-than-the-specified-maximum"] = True

    def _get_padding_truncation_strategies(self,
                                           padding=False,
                                           truncation=False,
                                           max_length=None,
                                           pad_to_multiple_of=None,
                                           verbose=True,
                                           **kwargs):
        """
        Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
        """
        old_truncation_strategy = kwargs.pop("truncation_strategy",
                                             "do_not_truncate")
        old_pad_to_max_length = kwargs.pop("pad_to_max_length", False)

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is False:
            if verbose:
                if not self.deprecation_warnings.get(
                        "Truncation-not-explicitly-activated", False):
                    logger.warning(
                        "Truncation was not explicitly activated but `max_length` is provided a specific value, please"
                        " use `truncation=True` to explicitly truncate examples to max length. Defaulting to"
                        " 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the"
                        " tokenizer you can select this strategy more precisely by providing a specific strategy to"
                        " `truncation`.")
                self.deprecation_warnings[
                    "Truncation-not-explicitly-activated"] = True
            truncation = "longest_first"

        # Get padding strategy
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn(
                    "The `pad_to_max_length` argument is deprecated and will be removed in a future version, "
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_length'` to pad to a max length. In this case, you can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the "
                    "maximal input size of the model (e.g. 512 for Bert).",
                    FutureWarning, )
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (
                            truncation is False or
                            truncation == "do_not_truncate"):
                        warnings.warn(
                            "`max_length` is ignored when `padding`=`True` and there is no truncation strategy. "
                            "To pad to max length, use `padding='max_length'`.")
                    if old_pad_to_max_length is not False:
                        warnings.warn(
                            "Though `pad_to_max_length` = `True`, it is ignored because `padding`=`True`."
                        )
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is False and old_truncation_strategy != "do_not_truncate":
            if verbose:
                warnings.warn(
                    "The `truncation_strategy` argument is deprecated and will be removed in a future version, use"
                    " `truncation=True` to truncate examples to a max length. You can give a specific length with"
                    " `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input"
                    " size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific"
                    " truncation strategy selected among `truncation='only_first'` (will only truncate the first"
                    " sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the"
                    " pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence"
                    " in the pairs).",
                    FutureWarning, )
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get(
                                "Asking-to-pad-to-max_length", False):
                            logger.warning(
                                "Asking to pad to max_length but no maximum length is provided and the model has no"
                                " predefined maximum length. Default to no padding."
                            )
                        self.deprecation_warnings[
                            "Asking-to-pad-to-max_length"] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get(
                                "Asking-to-truncate-to-max_length", False):
                            logger.warning(
                                "Asking to truncate to max_length but no maximum length is provided and the model has"
                                " no predefined maximum length. Default to no truncation."
                            )
                        self.deprecation_warnings[
                            "Asking-to-truncate-to-max_length"] = True
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (
                not self.pad_token or self.pad_token_id < 0):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and
                padding_strategy != PaddingStrategy.DO_NOT_PAD and
                pad_to_multiple_of is not None and max_length is not None and
            (max_length % pad_to_multiple_of != 0)):
            raise ValueError(
                "Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs

    def _pad(self,
             encoded_inputs,
             max_length=None,
             padding_strategy=PaddingStrategy.DO_NOT_PAD,
             pad_to_multiple_of=None,
             return_attention_mask=None):
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (
                max_length % pad_to_multiple_of != 0):
            max_length = (
                (max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(
            required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:

                    encoded_inputs["attention_mask"] = encoded_inputs[
                        "attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] +
                        [self.pad_token_type_id] * difference)
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs[
                        "special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[
                    0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [
                        0
                    ] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [
                        self.pad_token_type_id
                    ] * difference + encoded_inputs["token_type_ids"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[
                    0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(
                    self.padding_side))

        return encoded_inputs

    def pad(
            self,
            encoded_inputs,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_attention_mask=None,
            return_tensors=None,
            verbose=True, ):
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        `self.pad_token_id` and `self.pad_token_type_id`)

        <Tip>

        If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
                the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(
                encoded_inputs[0], Mapping):
            encoded_inputs = {
                key: [example[key] for example in encoded_inputs]
                for key in encoded_inputs[0].keys()
            }

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose)

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask, )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask, )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]` of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def _add_eos_if_not_present(self, token_ids):
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added.")
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def truncate_sequences(self,
                           ids,
                           pair_ids=None,
                           num_tokens_to_remove=0,
                           truncation_strategy="longest_first",
                           stride=0):
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (`int`, *optional*, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (`str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
                The strategy to follow for truncation. Can be:

                - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will truncate
                  token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a
                  batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater
                  than the model maximum admissible input size).
            stride (`int`, *optional*, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            `Tuple[List[int], List[int], List[int]]`: The truncated `ids`, the truncated `pair_ids` and the list of
            overflowing tokens. Note: The *longest_first* strategy returns empty list of overflowing tokens if a pair
            of sequences (or a batch of pairs) is provided.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.ONLY_FIRST or (
                truncation_strategy == TruncationStrategy.LONGEST_FIRST and
                pair_ids is None):
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                if self.truncation_side == "left":
                    overflowing_tokens = ids[:window_len]
                    ids = ids[num_tokens_to_remove:]
                elif self.truncation_side == "right":
                    overflowing_tokens = ids[-window_len:]
                    ids = ids[:-num_tokens_to_remove]
                else:
                    raise ValueError(
                        f"invalid truncation strategy: {self.truncation_side}, use 'left' or 'right'."
                    )

            else:
                error_msg = (
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the first sequence has a length {len(ids)}. ")
                if truncation_strategy == TruncationStrategy.ONLY_FIRST:
                    error_msg = (
                        error_msg +
                        "Please select another truncation strategy than "
                        f"{truncation_strategy}, for instance 'longest_first' or 'only_second'."
                    )
                logger.error(error_msg)
        elif truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            logger.warning(
                "Be aware, overflowing tokens are not returned for the setting you have chosen,"
                f" i.e. sequence pairs with the '{TruncationStrategy.LONGEST_FIRST.value}' "
                "truncation strategy. So the returned list will always be empty even if some "
                "tokens have been removed.")
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    if self.truncation_side == "right":
                        ids = ids[:-1]
                    elif self.truncation_side == "left":
                        ids = ids[1:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(
                            self.truncation_side))
                else:
                    if self.truncation_side == "right":
                        pair_ids = pair_ids[:-1]
                    elif self.truncation_side == "left":
                        pair_ids = pair_ids[1:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(
                            self.truncation_side))
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                if self.truncation_side == "right":
                    overflowing_tokens = pair_ids[-window_len:]
                    pair_ids = pair_ids[:-num_tokens_to_remove]
                elif self.truncation_side == "left":
                    overflowing_tokens = pair_ids[:window_len]
                    pair_ids = pair_ids[num_tokens_to_remove:]
                else:
                    raise ValueError("invalid truncation strategy:" + str(
                        self.truncation_side))
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the second sequence has a length {len(pair_ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    "for instance 'longest_first' or 'only_first'.")

        return (ids, pair_ids, overflowing_tokens)

    def num_special_tokens_to_add(self, pair: bool=False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def prepare_for_model(self,
                          ids,
                          pair_ids=None,
                          add_special_tokens=True,
                          padding=False,
                          truncation=False,
                          max_length=None,
                          stride=0,
                          pad_to_multiple_of=None,
                          return_tensors=None,
                          return_token_type_ids=None,
                          return_attention_mask=None,
                          return_overflowing_tokens=False,
                          return_special_tokens_mask=False,
                          return_offsets_mapping=False,
                          return_length=False,
                          verbose=True,
                          prepend_batch_axis=False,
                          **kwargs):
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different than `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs, )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None.")

        if (return_overflowing_tokens and
                truncation_strategy == TruncationStrategy.LONGEST_FIRST and
                pair_ids is not None):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`.")

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(
            pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride, )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(
                ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids)
                                               if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs[
                    "special_tokens_mask"] = self.get_special_tokens_mask(
                        ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(
            encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask, )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs,
            tensor_type=return_tensors,
            prepend_batch_axis=prepend_batch_axis)
        return batch_outputs

    def _batch_prepare_for_model(
            self,
            batch_ids_pairs,
            add_special_tokens=True,
            padding_strategy=PaddingStrategy.DO_NOT_PAD,
            truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
            max_length=None,
            stride=0,
            pad_to_multiple_of=None,
            return_tensors=None,
            return_token_type_ids=None,
            return_attention_mask=None,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_length=False,
            verbose=True, ):
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.
                value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose, )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask, )

        batch_outputs = BatchEncoding(
            batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    def _get_padding_truncation_strategies(self,
                                           padding=False,
                                           truncation=False,
                                           max_length=None,
                                           pad_to_multiple_of=None,
                                           verbose=True,
                                           **kwargs):
        """
        Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
        """
        old_truncation_strategy = kwargs.pop("truncation_strategy",
                                             "do_not_truncate")
        old_pad_to_max_length = kwargs.pop("pad_to_max_length", False)

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is False:
            if verbose:
                if not self.deprecation_warnings.get(
                        "Truncation-not-explicitly-activated", False):
                    logger.warning(
                        "Truncation was not explicitly activated but `max_length` is provided a specific value, please"
                        " use `truncation=True` to explicitly truncate examples to max length. Defaulting to"
                        " 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the"
                        " tokenizer you can select this strategy more precisely by providing a specific strategy to"
                        " `truncation`.")
                self.deprecation_warnings[
                    "Truncation-not-explicitly-activated"] = True
            truncation = "longest_first"

        # Get padding strategy
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn(
                    "The `pad_to_max_length` argument is deprecated and will be removed in a future version, "
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_length'` to pad to a max length. In this case, you can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the "
                    "maximal input size of the model (e.g. 512 for Bert).",
                    FutureWarning, )
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (
                            truncation is False or
                            truncation == "do_not_truncate"):
                        warnings.warn(
                            "`max_length` is ignored when `padding`=`True` and there is no truncation strategy. "
                            "To pad to max length, use `padding='max_length'`.")
                    if old_pad_to_max_length is not False:
                        warnings.warn(
                            "Though `pad_to_max_length` = `True`, it is ignored because `padding`=`True`."
                        )
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is False and old_truncation_strategy != "do_not_truncate":
            if verbose:
                warnings.warn(
                    "The `truncation_strategy` argument is deprecated and will be removed in a future version, use"
                    " `truncation=True` to truncate examples to a max length. You can give a specific length with"
                    " `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input"
                    " size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific"
                    " truncation strategy selected among `truncation='only_first'` (will only truncate the first"
                    " sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the"
                    " pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence"
                    " in the pairs).",
                    FutureWarning, )
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get(
                                "Asking-to-pad-to-max_length", False):
                            logger.warning(
                                "Asking to pad to max_length but no maximum length is provided and the model has no"
                                " predefined maximum length. Default to no padding."
                            )
                        self.deprecation_warnings[
                            "Asking-to-pad-to-max_length"] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get(
                                "Asking-to-truncate-to-max_length", False):
                            logger.warning(
                                "Asking to truncate to max_length but no maximum length is provided and the model has"
                                " no predefined maximum length. Default to no truncation."
                            )
                        self.deprecation_warnings[
                            "Asking-to-truncate-to-max_length"] = True
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (
                not self.pad_token or self.pad_token_id < 0):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and
                padding_strategy != PaddingStrategy.DO_NOT_PAD and
                pad_to_multiple_of is not None and max_length is not None and
            (max_length % pad_to_multiple_of != 0)):
            raise ValueError(
                "Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs

    def batch_encode_plus(self,
                          batch_text_or_text_pairs,
                          add_special_tokens=True,
                          padding=False,
                          truncation=False,
                          max_length=None,
                          stride=0,
                          is_split_into_words=False,
                          pad_to_multiple_of=None,
                          return_tensors=None,
                          return_token_type_ids=None,
                          return_attention_mask=None,
                          return_overflowing_tokens=False,
                          return_special_tokens_mask=False,
                          return_offsets_mapping=False,
                          return_length=False,
                          verbose=True,
                          **kwargs):
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            batch_text_or_text_pairs (`List[str]`, `List[Tuple[str, str]]`, `List[List[str]]`, `List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also `List[List[int]]`, `List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
                details in `encode_plus`).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs, )

        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs, )

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs,
            add_special_tokens=True,
            padding_strategy=PaddingStrategy.DO_NOT_PAD,
            truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
            max_length=None,
            stride=0,
            is_split_into_words=False,
            pad_to_multiple_of=None,
            return_tensors=None,
            return_token_type_ids=None,
            return_attention_mask=None,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_offsets_mapping=False,
            return_length=False,
            verbose=True,
            **kwargs):
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(
                            t, is_split_into_words=True, **kwargs)
                                          for t in text)))
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast.")

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0],
                                                        (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(
                pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose, )

        return BatchEncoding(batch_outputs)

    def tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """
        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        all_special_tokens_extended = dict(
            (str(t), t) for t in self.all_special_tokens_extended
            if isinstance(t, AddedToken))

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok)
                for s_tok in (self.unique_no_split_tokens +
                              self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern,
                          lambda m: m.groups()[0] or m.groups()[1].lower(),
                          text)

        no_split_token = set(self.unique_no_split_tokens)
        tokens = self.tokens_trie.split(text)
        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = all_special_tokens_extended.get(token, None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                else:
                    # We strip left and right by default
                    if right:
                        tokens[i + 1] = right.lstrip()
                    if left:
                        tokens[i - 1] = left.rstrip()
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text


class SPMTokenizer:
    r"""
    Constructs a tokenizer based on [SentencePiece](https://github.com/google/sentencepiece).

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """

    def __init__(self,
                 vocab_file,
                 split_by_punct=False,
                 sp_model_kwargs: Optional[Dict[str, Any]]=None):
        self.split_by_punct = split_by_punct
        self.vocab_file = vocab_file
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"{vocab_file} does not exist!")
        spm.load(vocab_file)
        bpe_vocab_size = spm.GetPieceSize()
        # Token map
        # <unk> 0+1
        # <s> 1+1
        # </s> 2+1
        self.vocab = {spm.IdToPiece(i): i for i in range(bpe_vocab_size)}
        self.ids_to_tokens = [spm.IdToPiece(i) for i in range(bpe_vocab_size)]
        # self.vocab['[PAD]'] = 0
        # self.vocab['[CLS]'] = 1
        # self.vocab['[SEP]'] = 2
        # self.vocab['[UNK]'] = 3

        self.spm = spm

    def __getstate__(self):
        state = self.__dict__.copy()
        state["spm"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        self.spm.Load(self.vocab_file)

    def tokenize(self, text):
        return self._encode_as_pieces(text)

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def decode(self, tokens, start=-1, end=-1, raw_text=None):
        if raw_text is None:
            return self.spm.decode_pieces([t for t in tokens])
        else:
            words = self.split_to_words(raw_text)
            word_tokens = [self.tokenize(w) for w in words]
            token2words = [0] * len(tokens)
            tid = 0
            for i, w in enumerate(word_tokens):
                for k, t in enumerate(w):
                    token2words[tid] = i
                    tid += 1
            word_start = token2words[start]
            word_end = token2words[end] if end < len(tokens) else len(words)
            text = "".join(words[word_start:word_end])
            return text

    def add_special_token(self, token):
        if token not in self.special_tokens:
            self.special_tokens.append(token)
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) - 1
                self.ids_to_tokens.append(token)
        return self.id(token)

    def part_of_whole_word(self, token, is_bos=False):
        if is_bos:
            return True
        if (len(token) == 1 and
            (_is_whitespace(list(token)[0]) or _is_control(list(token)[0]) or
             _is_punctuation(list(token)[0]))) or token in self.special_tokens:
            return False

        word_start = b"\xe2\x96\x81".decode("utf-8")
        return not token.startswith(word_start)

    def pad(self):
        return "[PAD]"

    def bos(self):
        return "[CLS]"

    def eos(self):
        return "[SEP]"

    def unk(self):
        return "[UNK]"

    def mask(self):
        return "[MASK]"

    def sym(self, id):
        return self.ids_to_tokens[id]

    def id(self, sym):
        return self.vocab[sym] if sym in self.vocab else 1

    def _encode_as_pieces(self, text):
        text = convert_to_unicode(text)
        if self.split_by_punct:
            words = self._run_split_on_punc(text)
            pieces = [self.spm.encode(w, out_type=str) for w in words]
            return [p for w in pieces for p in w]
        else:
            return self.spm.encode(text, out_type=str)

    def split_to_words(self, text):
        pieces = self._encode_as_pieces(text)
        word_start = b"\xe2\x96\x81".decode("utf-8")
        words = []
        offset = 0
        prev_end = 0
        for i, p in enumerate(pieces):
            if p.startswith(word_start):
                if offset > prev_end:
                    words.append(text[prev_end:offset])
                prev_end = offset
                w = p.replace(word_start, "")
            else:
                w = p
            try:
                s = text.index(w, offset)
                pn = ""
                k = i + 1
                while k < len(pieces):
                    pn = pieces[k].replace(word_start, "")
                    if len(pn) > 0:
                        break
                    k += 1

                if len(pn) > 0 and pn in text[offset:s]:
                    offset = offset + 1
                else:
                    offset = s + len(w)
            except Exception:
                offset = offset + 1

        if prev_end < offset:
            words.append(text[prev_end:offset])

        return words

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def save_pretrained(self, path: str, filename_prefix: str=None):
        filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
        if filename_prefix is not None:
            filename = filename_prefix + "-" + filename
        full_path = os.path.join(path, filename)
        with open(full_path, "wb") as fs:
            fs.write(self.spm.serialized_model_proto())
        return (full_path, )


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (
            cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError(f"Unsupported string type: {type(text)}")
