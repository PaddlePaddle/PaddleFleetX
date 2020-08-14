# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# There are 13 integer features and 26 categorical features
import os

sparse_feature_dim = 1000001
continous_features = range(1, 14)
categorial_features = range(14, 40)
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
class Dataset:
    def __init__(self):
        pass


class CriteoDataset(Dataset):
    def __init__(self, sparse_feature_dim):
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
        self.cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
        self.hash_dim_ = sparse_feature_dim
        # here, training data are lines with line_index < train_idx_
        self.train_idx_ = 41256555
        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)

    def process(self, file_list, directory):
        print(file_list)
        for file in file_list:
            content = []
	    with open(file, 'r') as f:
	        print("open file success")
	        line_idx = 0
	        for line in f:
		    line_idx += 1
		    features = line.rstrip('\n').split('\t')
		    dense_feature = []
		    sparse_feature = []
		    for idx in self.continuous_range_:
		        if features[idx] == '':
			    dense_feature.append(str(0.0))
		        else:
			    v = (float(features[idx]) - self.cont_min_[idx - 1]) / self.cont_diff_[idx - 1]
                            dense_feature.append(str(v))
                            
		    for idx in self.categorical_range_:
		        sparse_feature.append(str(hash(str(idx) + features[idx]) % self.hash_dim_))

                    line_new = [features[0]] + dense_feature + sparse_feature
		    content.append(line_new)
            with open(os.path.join(directory, os.path.basename(file)) + '.csv', 'w') as fout:
                for line in content:
                    #print(line)
                    fout.write(','.join(line))
                    fout.write('\n')

if __name__ == '__main__':
    dataset = CriteoDataset(sparse_feature_dim)
    raw_suffix = '_raw'
    processed_suffix = '_processed'
    directory = ['./train_data']
    for _ in directory:
        dir_ = _ + raw_suffix
        print("process %s" % dir_)
        file_list = os.listdir(dir_)
        file_list = [dir_ + '/' + file for file in file_list]
        dataset.process(file_list, _+processed_suffix)

