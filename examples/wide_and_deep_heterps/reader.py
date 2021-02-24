import paddle
import paddle.distributed.fleet as fleet
import os
import sys

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)


class WideDeepDatasetReader(fleet.MultiSlotDataGenerator):

    def line_process(self, line):
        features = line.rstrip('\n').split('\t')
        dense_feature = []
        sparse_feature = []
        for idx in continuous_range_:
            if features[idx] == "":
                dense_feature.append(0)
            else:
                dense_feature.append(
                    (float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1])
        for idx in categorical_range_:
            sparse_feature.append(
                [hash(str(idx) + features[idx]) % hash_dim_])
        label = [int(features[0])]
        return [dense_feature]+sparse_feature+[label]

    def generate_sample(self, line):
        def wd_reader():
            input_data = self.line_process(line)
            feature_name = ["dense_input"]
            for idx in categorical_range_:
                #feature_name.append("C" + str(idx - 13))
                feature_name.append(str(idx - 13))
            feature_name.append("label")
            yield zip(feature_name, input_data)

        return wd_reader

if __name__ == "__main__":
    my_data_generator = WideDeepDatasetReader()
    my_data_generator.set_batch(16)

    my_data_generator.run_from_stdin()
