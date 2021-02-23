config = dict()

#config['use_cvm'] = True
config['trainer'] = "PSGPUTrainer"
config['worker_class'] = "PSGPUWorker"
config['use_ps_gpu'] = True
config['worker_places'] = [0, 1, 2, 3, 4, 5, 6, 7]

#sparse_table config
sparse_config = dict()
sparse_config['sparse_table_class'] = "DownpourSparseTable"
sparse_config['sparse_accessor_class'] = "DownpourCtrAccessor"
config['embedding'] = sparse_config
config['pull_box_sparse_0.w_0'] = sparse_config

#dense_table config
dense_config = dict()
dense_config['dense_table_class'] = "DownpourDenseTable"
dense_config['dense_accessor_class'] = "DownpourDenseValueAccessor"
config['dense_table'] = dense_config

#data_norm_table config
datanorm_config = dict()
datanorm_config['datanorm_table_class'] = "DownpourDenseTable"
datanorm_config['datanorm_accessor_class'] = "DownpourDenseValueAccessor"
datanorm_config['datanorm_operation'] = "summary"
datanorm_config['datanorm_decay_rate'] = 0.999999
config['datanorm_table'] = datanorm_config
