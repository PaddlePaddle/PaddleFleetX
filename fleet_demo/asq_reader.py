import sys
import os
import paddle
import re
import collections
import time
import paddle.fluid.incubate.data_generator as dg

class MyDataset(dg.MultiSlotDataGenerator):

    def set_label_name(self, name):
        self.label_name = name

    def set_mod_value(self, value):
        self.mod_value = value

    def load_resource(self, dict_file):
        self._slot_dict = {}
        with open(dict_file) as fin:
            slots = fin.readlines()
        for i, slot in enumerate(slots):
            self._slot_dict[slot.strip()] = [False, i + 1]
        self.label_name = "label"
        self.mod_value = sys.maxsize

    def generate_sample(self, line):
        def data_iter():
            elements = line.split('\t')[0].split()[0:]
            padding = 0
            position_idx = -1
            label_value = int(elements[1])
            if label_value > 1:
                label_value = 1
            output = [(self.label_name, [label_value])]
            output += [(slot, []) for slot in self._slot_dict]
            for elem in elements[2:]:
                feasign, slot = elem.split(':')
                feasign = int(feasign) % self.mod_value
                if not self._slot_dict.has_key(slot):
                    continue
                self._slot_dict[slot][0] = True
                index = self._slot_dict[slot][1]
                output[index][1].append(feasign)
            for slot in self._slot_dict:
                visit, index = self._slot_dict[slot]
                if visit:
                    self._slot_dict[slot][0] = False
                else:
                    output[index][1].append(padding)
            yield output

        return data_iter

if __name__ == "__main__":
    d = MyDataset()
    arg_len = len(sys.argv)
    if arg_len < 2:
        print("Error: You need to define slot file")
        print("python asq_reader.py slotfile mod_value")
        sys.exit(-1)
    else:
        d.load_resource(sys.argv[1])
        if arg_len >= 3:
            d.set_mod_value(int(sys.argv[2]))
        d.run_from_stdin()
