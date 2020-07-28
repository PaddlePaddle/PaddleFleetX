import sys
import os
import paddle
import re
import collections
import numpy as np
import six
import time
import paddle.fluid.incubate.data_generator as dg
 
class DAMDataset(dg.MultiSlotStringDataGenerator):
    def normalize_length(self, _list, length, cut_type='tail'):
        """_list is a list or nested list, example turns/r/single turn c
           cut_type is head or tail, if _list len > length is used
           return a list len=length and min(read_length, length)
        """
        real_length = len(_list)
        if real_length == 0:
            return [0] * length, 0
    
        if real_length <= length:
            if not isinstance(_list[0], list):
                _list.extend([0] * (length - real_length))
            else:
                _list.extend([[]] * (length - real_length))
            return _list, real_length
    
        if cut_type == 'head':
            return _list[:length], length
        if cut_type == 'tail':
            return _list[-length:], length

    def produce_one_sample(self, y, c, r,
                           max_turn_num,
                           max_turn_len,
                           turn_cut_type='tail',
                           term_cut_type='tail'):
        """max_turn_num=10
           max_turn_len=50
           return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len
        """
        turns = c
        #normalize turns_c length, nor_turns length is max_turn_num
        nor_turns, turn_len = self.normalize_length(turns, max_turn_num, turn_cut_type)
    
        nor_turns_nor_c = []
        term_len = []
        #nor_turn_nor_c length is max_turn_num, element is a list length is max_turn_len
        for c in nor_turns:
            #nor_c length is max_turn_len
            nor_c, nor_c_len = self.normalize_length(c, max_turn_len, term_cut_type)
            nor_turns_nor_c.append(nor_c)
            term_len.append(nor_c_len)
    
        nor_r, r_len = self.normalize_length(r, max_turn_len, term_cut_type)
        return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len   

    def load_dict(self, dict_path, data_conf, source,
            turn_cut_type="tail", term_cut_type="tail"):
        self.turn_cut_type = turn_cut_type
        self.term_cut_type = term_cut_type
        self.data_conf = data_conf
        self.word2id = {}
        if source == "ubuntu":
            with open(dict_path) as f:
                word = None
                idx = 0
                for line in f:
                    x = line.strip()
                    if word is None:
                        word = x
                    else:
                        idx = int(x)
                        self.word2id[word] = idx
                        word = None
            self._eos = "_eos_"
            self._unk = "_unk_"
        elif source == "douban":
            with open(dict_path) as f:
                for line in f:
                    word, ids = line.strip().split('\t')
                    self.word2id[word] = int(ids)
            self._eos = "_EOS_"
            self._unk = "_OOV_"

    def generate_sample(self, line):
        def word_to_ids(words):
            return [self.word2id.get(
                w.replace('_', ''), self.word2id[self._unk])
                for w in words.strip().split(" ")]

        def data_iter():
            elements = line.strip().split('\t')
            y = int(elements[0])
            c = elements[1:-1]
            r = elements[-1]
            c2ids = []
            for words in c:
                ids = word_to_ids(words)
                c2ids.append(ids)
            r2ids = word_to_ids(r)

            y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len = self.produce_one_sample(
                    y, c2ids, r2ids, self.data_conf['max_turn_num'], self.data_conf['max_turn_len'],
                    self.turn_cut_type, self.term_cut_type)

            turns = np.array(nor_turns_nor_c).astype('int64')
            tt_turns_len = np.array(turn_len).astype('int64')
            every_turn_len = np.array(term_len).astype('int64')
            response = np.array(nor_r).astype('int64')
            response_len = np.array(r_len).astype('int64')
        
            max_turn_num = turns.shape[0]
            max_turn_len = turns.shape[1]
        
            turns_list = [turns[i, :] for i in six.moves.xrange(max_turn_num)]
            every_turn_len_list = [
                every_turn_len[i] for i in six.moves.xrange(max_turn_num)
            ]
        
            feed_list = []
            for i, turn in enumerate(turns_list):
                feed_list.append(("turn_{}".format(i),
                    [str(c) for c in turn]))
        
            for i, turn_len in enumerate(every_turn_len_list):
                turn_mask = np.ones(max_turn_len).astype("float32")
                turn_mask[turn_len:] = 0
                feed_list.append(("turn_mask_{}".format(i),
                    [str(x) for x in turn_mask]))
        
            feed_list.append(("response", 
                [str(x) for x in response]))
        
            response_mask = np.ones(max_turn_len).astype("float32")
            response_mask[response_len:] = 0
            feed_list.append(("response_mask",
                [str(x) for x in response_mask]))
        
            feed_list.append(("label", [str(float(y))]))
            yield feed_list
        return data_iter
        
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: cat <data> | python data_generator.py <dict> <max_turn_num> <max_turn_len> <source>")
        exit(-1)

    dict_path = sys.argv[1]
    max_turn_num = int(sys.argv[2])
    max_turn_len = int(sys.argv[3])
    source = sys.argv[4]
    data_conf = {
        "max_turn_num": max_turn_num,
        "max_turn_len": max_turn_len,
    }
    a = DAMDataset()
    a.load_dict(dict_path, data_conf, source)
    a.run_from_stdin()
