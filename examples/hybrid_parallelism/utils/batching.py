#coding=utf8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Mask, padding and batching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import copy
import utils.word_order as word_order
import collections

import six
from six.moves import xrange

rng = random.Random(12345)


def mask(args,
         batch_tokens,
         seg_labels,
         mask_word_tags,
         total_token_num,
         vocab_size,
         CLS,
         SEP,
         MASK,
         masking_strategy):

    #return batch_tokens, [[21]], [[4]], [], []
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    max_len = max([len(sent) for sent in batch_tokens])
    mask_label = []
    mask_pos = []
    prob_mask = np.random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    # TODO: think a way to protect CLS SEP MASK PAD
    replace_ids = np.random.randint(3, high=vocab_size, size=total_token_num)
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        mask_pos.append([])
        mask_flag = False
        mask_word = mask_word_tags[sent_index]
        prob_index += pre_sent_len
        if mask_word and masking_strategy != 'random_masking':
            beg = 0
            for token_index, token in enumerate(sent):
                seg_label = seg_labels[sent_index][token_index]
                # TODO fix the hard coding
                if seg_label == 1:
                    continue
                if beg == 0:
                    if seg_label != -1:
                        beg = token_index
                    continue

                prob = prob_mask[prob_index + beg]
                if prob > 0.12:
                    pass
                else:
                    for index in xrange(beg, token_index):
                        prob = random.random()
                        base_prob = 1.0
                        if 0.2 < prob:
                            mask_label.append(sent[index])
                            sent[index] = MASK
                            mask_flag = True
                            mask_pos[-1].append(index)
                        elif 0.1 < prob <= 0.2:
                            mask_label.append(sent[index])
                            sent[index] = replace_ids[prob_index + index]
                            mask_flag = True
                            mask_pos[-1].append(index)
                        else:
                            mask_label.append(sent[index])
                            mask_pos[-1].append(index)

                if seg_label == -1:
                    beg = 0
                else:
                    beg = token_index
        else:
            for token_index, token in enumerate(sent):
                if token == CLS:
                    continue
                prob = prob_mask[prob_index + token_index]
                if prob > 0.15:
                    continue
                elif 0.03 < prob <= 0.15:
                    # mask
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        sent[token_index] = MASK
                        mask_flag = True
                        mask_pos[-1].append(token_index)
                elif 0.015 < prob <= 0.03:
                    # random replace
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        sent[token_index] = replace_ids[prob_index +
                                                        token_index]
                        mask_flag = True
                        mask_pos[-1].append(token_index)
                else:
                    # keep the original token
                    if token != SEP and token != CLS:
                        mask_label.append(sent[token_index])
                        mask_pos[-1].append(token_index)

        pre_sent_len = len(sent)

    mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])

    wordorder_batch_tokens, word_order_label, word_order_pos \
            = word_order.word_reorder(batch_tokens, mask_pos, MASK, SEP, CLS, rng)
    if args.word_order:
         batch_tokens = wordorder_batch_tokens

    flatten_mask_pos = []
    for sent_index in range(len(mask_pos)):
        for p in mask_pos[sent_index]:
            flatten_mask_pos.append(sent_index * max_len + p)
    mask_pos = np.array(flatten_mask_pos).astype("int64").reshape([-1, 1])

    word_order_label = np.array(word_order_label).astype('int64').reshape([-1, 1])

    return batch_tokens, mask_label, mask_pos, word_order_label, word_order_pos


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def _is_start_piece_sp(piece):
  """Check if the current word piece is the starting piece (sentence piece)."""
  special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
  special_pieces.add(u"€".encode("utf-8"))
  special_pieces.add(u"£".encode("utf-8"))
  # Note(mingdachen):
  # For foreign characters, we always treat them as a whole piece.
  english_chars = set(list("abcdefghijklmnopqrstuvwhyz"))
  if (six.ensure_str(piece).startswith("▁") or
      six.ensure_str(piece).startswith("<") or piece in special_pieces or
      not all([i.lower() in english_chars.union(special_pieces)
               for i in piece])):
    return True
  else:
    return False


def _is_start_piece_bert(piece):
  """Check if the current word piece is the starting piece (BERT)."""
  # When a word has been split into
  # WordPieces, the first token does not have any marker and any subsequence
  # tokens are prefixed with ##. So whenever we see the ## token, we
  # append it to the previous set of word indexes.
  return not six.ensure_str(piece).startswith("##")


def is_start_piece(piece, using_spm=True):
  if using_spm:
    return _is_start_piece_sp(piece)
  else:
    return _is_start_piece_bert(piece)


def times(arr, normalize=False):
    t = 1.0
    out = []
    for e in arr:
        t = t * e
        out.append(t)
    out = np.array(out)
    if normalize:
        out = out / out.sum(keepdims=True)
    return out


def create_masked_lm_predictions(tokens, using_spm, ngram, do_whole_word_mask, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, favor_longer_ngram=True, do_permutation=False):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and not is_start_piece(token, using_spm)):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(token, using_spm):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                        masked_lm_labels, token_boundary)

    num_to_predict = min(max_predictions_per_seq,
                                             max(1, int(round(len(tokens) * masked_lm_prob))))
    #num_to_predict = max(1, int(round(len(tokens) * masked_lm_prob)))

    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, ngram + 1)
    pvals /= pvals.sum(keepdims=True)

    if favor_longer_ngram:
        pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx+n])
        ngram_indexes.append(ngram_index)

    rng.shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np.random.choice(ngrams[:len(cand_index_set)],
                                                 p=pvals[:len(cand_index_set)] /
                                                 pvals[:len(cand_index_set)].sum(keepdims=True))
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict

    rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                                     p=pvals[:len(cand_index_set)] /
                                                     pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)


def n_gram_masking(tokenizer,
                   batch_ids,
                   vocab_size,
                   ngram,
                   CLS,
                   SEP,
                   MASK,
                   masked_lm_prob,
                   max_predictions_per_seq,
                   rng,
                   using_spm,
                   do_whole_word_mask,
                   connect_prob,
                   using_connective_masking=False,
                   masking_strategy=None,
                    ):

    vocab_words = list(tokenizer.vocab.keys())
    max_len = max([len(sent) for sent in batch_ids])

    mask_label = []
    mask_pos = []
    new_batch_ids = []

    for ith, (ids, conn_prob) in enumerate(zip(batch_ids, connect_prob)):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        if masking_strategy == "ngram_masking":
            output_tokens, masked_lm_positions, masked_lm_labels, token_boundary = \
                   create_masked_lm_predictions(tokens, \
                                                using_spm,
                                                ngram,
                                                do_whole_word_mask,
                                                masked_lm_prob, \
                                                max_predictions_per_seq, \
                                                vocab_words, \
                                                rng)
        else:
            raise ValueError('unsupported masking strategy')

        ids = tokenizer.convert_tokens_to_ids(output_tokens)
        new_batch_ids.append(ids)

        label = tokenizer.convert_tokens_to_ids(masked_lm_labels)
        mask_label.extend(label)

        mask_pos.extend([ith*max_len + p for p in masked_lm_positions])

    mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])

    batch_ids, word_order_label, word_order_pos = word_order.word_reorder(new_batch_ids, mask_pos, MASK, SEP, CLS, rng)
    word_order_label = np.array(word_order_label).astype('int64').reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype('int64').reshape([-1, 1])

    return batch_ids, mask_label, mask_pos, word_order_label, word_order_pos


def prepare_batch_data(insts,
                       total_token_num,
                       task_index,
                       lm_weight,
                       task_num,
                       tokenizer=None,
                       voc_size=0,
                       pad_id=None,
                       cls_id=None,
                       sep_id=None,
                       mask_id=None,
                       return_input_mask=True,
                       return_max_len=True,
                       return_num_token=False,
                       args=None,
                       ):

    batch_src_ids = [inst[0] for inst in insts]
    batch_sent_ids = [inst[1] for inst in insts]
    batch_pos_ids = [inst[2] for inst in insts]
    batch_task_ids = [inst[3] for inst in insts]
    labels = [inst[4] for inst in insts]
    labels = np.array(labels).astype("int64").reshape([-1, 1])
    seg_labels = [inst[5] for inst in insts]
    case_labels = [inst[6] for inst in insts]
    appear_labels = [inst[7] for inst in insts]
    appear_switch = [inst[8] for inst in insts]
    mask_word_tags = [inst[9] for inst in insts]

    # First step: do mask without padding
    assert mask_id >= 0, "[FATAL] mask_id must >= 0"

    not_mask = False
    if lm_weight < 0.01:
        not_mask = True

    if args.masking_strategy in ['ngram_masking']:
        connect_prob = seg_labels
        # TODO 512/0.15 hard code from google/bert, should fix
        masked_lm_prob = 0.15
        out, mask_label, mask_pos, word_order_label, word_order_pos = n_gram_masking(
                   tokenizer, 
                   batch_src_ids,
                   vocab_size=voc_size,
                   ngram=args.ngram,
                   CLS=cls_id,
                   SEP=sep_id,
                   MASK=mask_id,
                   masked_lm_prob=masked_lm_prob,
                   max_predictions_per_seq=int(512 * masked_lm_prob),
                   rng=rng, 
                   using_spm=args.using_spm,
                   do_whole_word_mask=args.do_whole_word_mask,
                   masking_strategy=args.masking_strategy,
                   connect_prob=connect_prob)

    elif args.masking_strategy in ['connective_chunk_masking', 'chunk_masking', 'random_masking']:
        if not_mask:
            out = copy.deepcopy(batch_src_ids)
            _, mask_label, mask_pos, word_order_label, word_order_pos = mask(
                args,
                batch_src_ids,
                seg_labels,
                mask_word_tags,
                total_token_num,
                vocab_size=voc_size,
                CLS=cls_id,
                SEP=sep_id,
                MASK=mask_id,
                masking_strategy=args.masking_strategy)
        else:
            out, mask_label, mask_pos, word_order_label, word_order_pos = mask(
                args,
                batch_src_ids,
                seg_labels,
                mask_word_tags,
                total_token_num,
                vocab_size=voc_size,
                CLS=cls_id,
                SEP=sep_id,
                MASK=mask_id,
                masking_strategy=args.masking_strategy)
    else:
        raise ValueError('unsupported masking strategy')

    # Second step: padding
    src_id, self_input_mask, seq_lens = pad_batch_data(out, pad_idx=pad_id, return_input_mask=True, return_seq_lens=True)
    pos_id = pad_batch_data(batch_pos_ids, pad_idx=pad_id)
    sent_id = pad_batch_data(batch_sent_ids, pad_idx=pad_id)
    task_id = pad_batch_data(batch_task_ids, pad_idx=pad_id)

    seg_labels = pad_batch_data(seg_labels, pad_idx=pad_id)
    case_labels = pad_batch_data(case_labels, pad_idx=pad_id)
    appear_labels = pad_batch_data(appear_labels, pad_idx=pad_id)

    lm_w = np.array([lm_weight]).astype("float32")

    return_list = [
        src_id, pos_id, sent_id, task_id, self_input_mask, mask_label, mask_pos,
        word_order_label, word_order_pos, lm_w,
        case_labels, appear_labels, appear_switch[0], seq_lens
    ]

    for i in xrange(task_num):
        if i == task_index:
            return_list.append(labels)
            return_list.append(np.array([1.0]).astype("float32"))
        else:
            return_list.append(np.zeros_like(labels))
            return_list.append(np.array([0.0]).astype("float32"))

    #print(return_list)
    #print('src_id',src_id)
    #print('pos_id',pos_id)
    #print('self_input_mask',self_input_mask)
    #print('mask_label',mask_label)
    #print('mask_pos',mask_pos)

    return return_list


def pad_batch_data(insts,
                   pad_idx=0,
                   max_seq_len=None, 
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts) if max_seq_len is None else max_seq_len
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    #print(insts)

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts]
        )
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] *
                                    (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


if __name__ == "__main__":
    pass
