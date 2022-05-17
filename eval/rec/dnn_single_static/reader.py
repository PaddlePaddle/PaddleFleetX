import paddle
import os
import numpy as np
from paddle.io import IterableDataset

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)


class DnnDataset(IterableDataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def line_process(self, line):
        features = line.rstrip('\n').split('\t')
        dense_feature = []
        sparse_feature = []
        for idx in continuous_range_:
            if features[idx] == "":
                dense_feature.append(0.0)
            else:
                dense_feature.append(
                    (float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1])
        for idx in categorical_range_:
            sparse_feature.append(
                [hash(str(idx) + features[idx]) % hash_dim_])
        label = [int(features[0])]
        output_list = []
        output_list.append(np.array(dense_feature).astype('float32'))
        for sparse in sparse_feature:
            output_list.append(np.array(sparse).astype('int64'))
        output_list.append(np.array(label).astype('int64'))
        return output_list

    def __iter__(self):
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    input_data = self.line_process(line)
                    yield input_data