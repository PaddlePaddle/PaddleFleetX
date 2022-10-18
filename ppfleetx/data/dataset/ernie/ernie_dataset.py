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

import os
import sys
import time
import numpy as np
import re
import copy
import paddle

from .dataset_utils import (
    get_samples_mapping,
    get_a_and_b_segments,
    truncate_segments,
    create_tokens_and_tokentypes,
    create_masked_lm_predictions,
    make_indexed_dataset,
    get_indexed_dataset_, )
from paddlenlp.transformers import ErnieTokenizer


def get_local_rank():
    return int(os.getenv("PADDLE_RANK_IN_NODE", 0))


print_rank_0 = print

mode_to_index = {"Train": 0, "Eval": 1, "Test": 2}


class ErnieDataset(paddle.io.Dataset):
    def __init__(self, input_dir, tokenizer_type, split, num_samples, mode,
                 max_seq_length, masked_lm_prob, short_seq_prob, seed,
                 binary_head, share_folder, favor_longer_ngram, max_ngrams):
        tokenizer = ErnieTokenizer.from_pretrained(tokenizer_type)
        tokenizer.extend_chinese_char()

        # print("input_dir", input_dir)
        files = get_train_data_file(input_dir)[0]
        # print("files", files)
        # assert len(files) == 1
        skip_warmup = True
        indexed_dataset = get_indexed_dataset_(files, None, skip_warmup)
        total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1
        splits = get_train_valid_test_split_(split, total_num_of_documents)
        # Print stats about the splits.
        print_rank_0(' > dataset split:')

        def print_split_stats(name, index):
            print_rank_0('    {}:'.format(name))
            print_rank_0('     document indices in [{}, {}) total of {} '
                         'documents'.format(splits[index], splits[index + 1],
                                            splits[index + 1] - splits[index]))
            start_index = indexed_dataset.doc_idx[splits[index]]
            end_index = indexed_dataset.doc_idx[splits[index + 1]]
            print_rank_0('     sentence indices in [{}, {}) total of {} '
                         'sentences'.format(start_index, end_index, end_index -
                                            start_index))

        index = mode_to_index[mode]
        print_split_stats(mode, index)

        # dataset = None
        assert splits[index + 1] > splits[index]
        # Get the pointer to the original doc-idx so we can set it later.
        doc_idx_ptr = indexed_dataset.get_doc_idx()
        # Slice the doc-idx
        start_index = splits[index]
        # Add +1 so we can index into the dataset to get the upper bound.
        end_index = splits[index + 1] + 1
        # New doc_idx view.
        indexed_dataset.set_doc_idx(doc_idx_ptr[start_index:end_index])
        # Build the dataset accordingly.
        # kwargs = dict(
        #     name=name,
        #     data_prefix=data_prefix,
        #     num_epochs=None,
        #     max_num_samples=train_valid_test_num_samples[index],
        #     max_seq_length=max_seq_length,
        #     seed=seed,
        #     share_folder=args.share_folder,
        #     args=args,
        # )
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.binary_head = binary_head
        self.share_folder = share_folder
        self.indexed_dataset = indexed_dataset

        self.favor_longer_ngram = favor_longer_ngram
        self.max_ngrams = max_ngrams

        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping(
            self.indexed_dataset,
            files,
            None,
            num_samples,
            self.max_seq_length - 3,  # account for added tokens
            short_seq_prob,
            self.seed,
            mode,
            self.binary_head,
            self.share_folder)

        # print_split_stats('validation', 1)
        # print_split_stats('test', 2)  

        # train_valid_test_num_samples = [
        #     args.global_batch_size * args.max_steps,
        #     args.micro_batch_size * (args.max_steps // args.eval_freq + 1) *
        #     args.eval_iters * data_world_size,
        #     args.micro_batch_size * args.test_iters * data_world_size]

        # def __init__(
        #         self,
        #         name,
        #         tokenizer,
        #         indexed_dataset,
        #         data_prefix,
        #         num_epochs,
        #         max_num_samples,
        #         masked_lm_prob,
        #         max_seq_length,
        #         short_seq_prob,
        #         seed,
        #         binary_head,
        #         share_folder=False,
        #         args=None, ):

        #     # Params to store.
        #     self.name = name
        #     self.seed = seed
        #     self.masked_lm_prob = masked_lm_prob
        #     self.max_seq_length = max_seq_length
        #     self.binary_head = binary_head
        #     self.share_folder = share_folder
        #     self.args = args

        #     # Dataset.
        #     self.indexed_dataset = indexed_dataset

        #     # Build the samples mapping.
        #     self.samples_mapping = get_samples_mapping(
        #         self.indexed_dataset,
        #         data_prefix,
        #         num_epochs,
        #         max_num_samples,
        #         self.max_seq_length - 3,  # account for added tokens
        #         short_seq_prob,
        #         self.seed,
        #         self.name,
        #         self.binary_head,
        #         self.share_folder)

        # Vocab stuff.
        # tokenizer = get_tokenizer()
        # self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        # self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.vocab_id_list = list(tokenizer.vocab.idx_to_token.keys())
        self.vocab_id_to_token_dict = copy.deepcopy(
            tokenizer.vocab.idx_to_token)
        self.vocab_token_to_id_dict = copy.deepcopy(
            tokenizer.vocab.token_to_idx)

        # ERNIE is chinse char level model, sometime is need
        # add ## chinse char to encode and decode.
        # Here we extend the vocab dict.
        self.vocab_id_to_token_dict.update(tokenizer.added_tokens_decoder)
        self.vocab_token_to_id_dict.update(tokenizer.added_tokens_encoder)

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):

        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]

        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2**32))
        return build_training_sample(
            sample,
            seq_length,
            self.max_seq_length,  # needed for padding
            self.vocab_id_list,
            self.vocab_id_to_token_dict,
            self.vocab_token_to_id_dict,
            self.cls_id,
            self.sep_id,
            self.mask_id,
            self.pad_id,
            self.masked_lm_prob,
            np_rng,
            self.binary_head,
            self.favor_longer_ngram,
            self.max_ngrams)


def build_training_sample(sample,
                          target_seq_length,
                          max_seq_length,
                          vocab_id_list,
                          vocab_id_to_token_dict,
                          vocab_token_to_id_dict,
                          cls_id,
                          sep_id,
                          mask_id,
                          pad_id,
                          masked_lm_prob,
                          np_rng,
                          binary_head,
                          favor_longer_ngram=False,
                          max_ngrams=3):
    """Biuld training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        vocab_token_to_id_dict: A dictionary from text tokens to vocab ids.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
    """

    if binary_head:
        # We assume that we have at least two sentences in the sample
        assert len(sample) > 1, "The sentence num should be large than 1."
    assert target_seq_length <= max_seq_length

    # Divide sample into two segments (A and B).
    if binary_head:
        tokens_a, tokens_b, is_next_random = get_a_and_b_segments(sample,
                                                                  np_rng)
    else:
        tokens_a = []
        for j in range(len(sample)):
            tokens_a.extend(sample[j])
        tokens_b = []
        is_next_random = False

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncated = truncate_segments(tokens_a, tokens_b,
                                  len(tokens_a),
                                  len(tokens_b), max_num_tokens, np_rng)

    # Build tokens and toketypes.
    tokens, tokentypes = create_tokens_and_tokentypes(tokens_a, tokens_b,
                                                      cls_id, sep_id)

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _,
     _) = create_masked_lm_predictions(
         tokens,
         vocab_id_list,
         vocab_id_to_token_dict,
         masked_lm_prob,
         cls_id,
         sep_id,
         mask_id,
         max_predictions_per_seq,
         np_rng,
         vocab_token_to_id_dict=vocab_token_to_id_dict,
         to_chinese_char=True,
         inplace_random_mask=False,
         favor_longer_ngram=favor_longer_ngram,
         max_ngrams=max_ngrams, )

    # Padding.
    tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np \
        = pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                                   masked_labels, pad_id, max_seq_length)

    return tokens_np, tokentypes_np, padding_mask_np, masked_positions, masked_labels, int(
        is_next_random)


def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                             masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array(
        [1] * num_tokens + [0] * padding_length, dtype=np.float32)
    padding_mask_np = (1 - padding_mask_np) * -1e4

    padding_mask_np = padding_mask_np.reshape([1, 1, -1])
    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np


def get_train_data_file(input_dir):
    if len(input_dir.split()) > 1:
        # weight-1 data-prefix-1 weight-2 data-prefix-2 ...
        return input_dir.split()
    else:
        files = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if (os.path.isfile(os.path.join(input_dir, f)) and "_idx.npz" in
                str(f))
        ]
        # print(">>>> files", files)
        files = [x.replace("_idx.npz", "") for x in files]

        if len(files) > 1:
            ret = []
            logger.info("You are using multi-dataset:")
            for x in files:
                ret.append(1.0)
                ret.append(x)
                logger.info("    > set weight of %s dataset to 1.0" % x)
            return ret
    return files


def get_train_valid_test_split_(splits, size):
    """
    Get dataset splits from comma or '/' separated string list.
    """

    splits = [float(s) for s in splits]
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(
            round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index
