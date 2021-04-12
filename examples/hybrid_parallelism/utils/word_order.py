#!coding=utf8

import random
import numpy as np


def shuffle_gram(gram):
    if len(set(gram)) <= 1:
        return gram

    new_gram = gram[:]
    #while new_gram == gram:
    random.shuffle(new_gram)
    return new_gram


def word_reorder_sent(sent, mask_id, sep_id, prob_t):
    new_sent = sent[:]
    n_gram = 3
    i = 1
    
    label = []
    pos = []
    if len(sent[i:]) < n_gram:
        return sent, label, pos

    while i < len(new_sent):

        substrs = new_sent[i:i+n_gram]

        # 满足条件 shuffle gram
        if len(substrs) == n_gram and (mask_id not in substrs) and (sep_id not in substrs) and (random.random() < prob_t):
            label.extend(substrs)
            pos.extend(list(range(i, i + n_gram)))
            for i, t in zip(range(i, i + n_gram), shuffle_gram(substrs)):
            #for i, t in zip(range(i, i + n_gram), substrs):
                new_sent[i] = t
            i += n_gram
        else:
            i += 1

    return new_sent, label, pos


def word_reorder_sent_v2(sent, mask_indexes, mask_id, sep_id, cls_id, prob_t, rng):
    """ word reorder 策略重写(相对于word_reorder_sent) """
    output_tokens = list(sent)
    num_to_predict = prob_t * len(sent)

    mask_indexes_set = set(mask_indexes)

    ngram_indexes = [] 
    n = 3
    cand_indexes = []
    for (i, token_id) in enumerate(sent):
        ngram_tmp = sent[i: i + n]
        indexes = list(range(i, i + n))
        if cls_id in ngram_tmp or mask_id in ngram_tmp or sep_id in ngram_tmp:
            continue

        if set(indexes) & set(mask_indexes_set):
            continue

        if len(ngram_tmp) != n:
            continue
        ngram_indexes.append(indexes)

    # TODO  prob_t * ngram_indexes ?
    rng.shuffle(ngram_indexes)

    reorder_lm_labels = []
    reorder_lm_positions = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(reorder_lm_labels) >= num_to_predict:
            break

        is_any_index_covered = False
        for index in cand_index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue

        new_index_set = shuffle_gram(cand_index_set)
        for i, j in zip(cand_index_set, new_index_set):
            covered_indexes.add(index)
            output_tokens[i] = sent[j] # shuffle gram
            reorder_lm_positions.append(i)
            reorder_lm_labels.append(sent[i])

    return output_tokens, reorder_lm_labels, reorder_lm_positions


def safe_guard(batch_tokens, mask_id):
    labels = [batch_tokens[0][1]]
    labels_pos = [1]

    return batch_tokens, labels, labels_pos


def word_reorder(batch_tokens, mask_pos, MASK, SEP, CLS, rng):
    """ word reorder 策略主接口 """
    assert len(np.squeeze(batch_tokens)) != 0, batch_tokens

    max_len = max([len(sent) for sent in batch_tokens])
    labels = []
    labels_pos = []
    new_batch_tokens = []

    prob_t = 0.03

    for sent_index, (sent, mask_indexes) in enumerate(zip(batch_tokens, mask_pos)):
        new_sent, label, pos = word_reorder_sent_v2(sent, mask_indexes, MASK, SEP, CLS, prob_t, rng)

        new_batch_tokens.append(new_sent)
        labels.extend([[l] for l in label])
        labels_pos.extend([[i + sent_index * max_len] for i in pos])

    return new_batch_tokens, labels, labels_pos    


#batch = []
#
#for i, line in enumerate(open('part-00387')):
#    line = line.strip()
#    arr = line.split(';')
#    toks = arr[0].split(' ')
#
#    for j, _ in enumerate(toks):
#        toks[j] = int(toks[j])
#        if random.random() < 0.15:
#            toks[j] = 3
#
#    batch.append(toks)
#    if len(batch) > 128:
#        print(i)
#        if i == []:
#            import pdb
#            pdb.set_trace()
#        word_reorder(batch)
#        batch = []
#

#print(word_reorder([[1,2,3,4,5,6,7,8,9], [6,7,8,9]], 3))
#print(word_reorder([[1,9], [6,9]], 3))

#import numpy as np
#new_batch_tokens, labels, labels_pos = word_reorder([[1,2,3,4,5,6,7,8,9], [6,7,8,9]], 3)
#
#print(labels)
#print(labels_pos)
#print(np.array(labels).astype('int32').reshape([-1, 1]))
#print(np.array(labels_pos).astype('int32').reshape([-1, 1]))

# print(word_reorder_sent([1,2,3,4,5,6,7,8,9], 3, 0.5))



