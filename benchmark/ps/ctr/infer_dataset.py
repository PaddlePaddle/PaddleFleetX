import argparse
import logging

import numpy as np
# disable gpu training for this example 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import paddle
import paddle.fluid as fluid

import criteo_reader
from network_conf import ctr_dnn_model_dataset


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle DeepFM example")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="The path of model parameters gz file")
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="The path of the dataset to infer")
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help="The size for embedding layer (default:10)")
    parser.add_argument(
        '--sparse_feature_dim',
        type=int,
        default=1000001,
        help="The size for embedding layer (default:1000001)")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help="The size of mini-batch (default:1000)")

    return parser.parse_args()

def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

def data2tensor(data, place):
    feed_dict = {}
    dense = data[0]
    sparse = data[1:-1]
    y = data[-1]
    dense_data = np.array([x[0] for x in data]).astype("float32")
    dense_data = dense_data.reshape([-1, 13])
    feed_dict["dense_input"] = dense_data
    for i in range(26):
        sparse_data = to_lodtensor([x[1 + i] for x in data], place)
        feed_dict["C" + str(1 + i)] = sparse_data
    
    y_data = np.array([x[-1] for x in data]).astype("int64")
    y_data = y_data.reshape([-1, 1])
    feed_dict["label"] = y_data
    return feed_dict

def infer():
    args = parse_args()

    place = fluid.CPUPlace()
    inference_scope = fluid.core.Scope()

    test_files = ["%s/%s" % (args.data_path, x) for x in os.listdir(args.data_path)]
    from criteo_reader import CriteoDataset
    criteo_dataset = CriteoDataset()
    criteo_dataset.setup(args.sparse_feature_dim)
    exe = fluid.Executor(place)
    
    train_thread_num = 10
    
    def set_zero(var_name):
        param = inference_scope.var(var_name).get_tensor()
        param_array = np.zeros(param._get_dims()).astype("int64")
        param.set(param_array, place)

    epochs = 20
    epoch_list = range(20)
    for i in epoch_list:
        cur_model_path = args.model_path + "/epoch" + str(i + 1) + ".model"
        with fluid.scope_guard(inference_scope):
            [inference_program, feed_target_names, fetch_targets] = \
                        fluid.io.load_inference_model(cur_model_path, exe)
            input_vars = [inference_program.global_block().var(name) for name in feed_target_names]
            dataset = fluid.DatasetFactory().create_dataset()
            dataset.set_use_var(input_vars)
            pipe_command = "python criteo_reader.py %d" % args.sparse_feature_dim
            dataset.set_pipe_command(pipe_command)
            dataset.set_batch_size(100)
            dataset.set_filelist(test_files)
            dataset.set_thread(10)
            auc_states_names = ['_generated_var_2', '_generated_var_3']
            for name in auc_states_names:
                set_zero(name)
            #test_reader = criteo_dataset.infer_reader(test_files, 1000, 100000)
            #for batch_id, data in enumerate(test_reader()):
            '''
            loss_val, auc_val = exe.run(inference_program,
            feed=data2tensor(data, place),
            fetch_list=fetch_targets,
            use_program_cache=True)
            '''
            print(dataset.desc())
            exe.infer_from_dataset(program=inference_program,
                                   dataset=dataset,
                                   fetch_list=[fetch_targets[0]],
                                   fetch_info=["auc"])
            #print("train_pass_%d, test_pass_%d\t%f" % (i - 1, i, auc_val))


if __name__ == '__main__':
    infer()
