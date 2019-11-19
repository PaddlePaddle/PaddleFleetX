import os

dict_size = 354051             
embedding_size = 300
neg_num = 5
window_size = 5
batch_size = 100
infer_batch_size = 2000
num_epochs = 10
learning_rate = 1.0
decay_steps = 100000
decay_rate = 0.999
cpu_num = int(os.getenv("CPU_NUM", "1"))

train_files_path = "./train_data/"
test_files_path = "./test_data/"
dict_path = "./thirdparty/test_build_dict"
infer_dict_path = "./thirdparty/test_build_dict_word_to_id_"
model_path = "./model"
if not os.path.exists(model_path):
    os.mkdir(model_path)

is_local_cluster = True
is_geo_sgd = True

