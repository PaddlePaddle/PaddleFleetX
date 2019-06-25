import infer_args
import logging

import numpy as np
# disable gpu training for this example 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import paddle
import paddle.fluid as fluid
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def infer():
    args = infer_args.parse_args()

    place = fluid.CPUPlace()
    inference_scope = fluid.core.Scope()

    test_files = ["%s/%s" % (args.data_path, x) for x in os.listdir(args.data_path)]
    exe = fluid.Executor(place)
    train_thread_num = 10
    
    def set_zero(var_name):
        param = inference_scope.var(var_name).get_tensor()
        param_array = np.zeros(param._get_dims()).astype("int64")
        param.set(param_array, place)

    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = \
                    fluid.io.load_inference_model(args.model_path, exe)
        input_vars = [inference_program.global_block().var(name) for name in feed_target_names]
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(input_vars)
        pipe_command = "python criteo_reader.py %d" % args.sparse_feature_dim
        dataset.set_pipe_command(pipe_command)
        dataset.set_batch_size(1000)
        dataset.set_filelist(test_files)
        dataset.set_thread(10)
        auc_states_names = ['_generated_var_2', '_generated_var_3']
        for name in auc_states_names:
            set_zero(name)
        exe.infer_from_dataset(program=inference_program,
                               dataset=dataset,
                               fetch_list=[fetch_targets[1]],
                               fetch_info=["auc"],
                               print_period=10)

if __name__ == '__main__':
    infer()
