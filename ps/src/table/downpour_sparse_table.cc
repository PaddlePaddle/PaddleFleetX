#include "table/downpour_sparse_table.h"
#include "common/registerer.h"
#include "common/matrix.h"
#include "Eigen/Dense"

DEFINE_bool(pslib_print_missed_key_num_every_push, false, "print push sparse key missed num");
DEFINE_bool(pslib_create_value_when_push, false, "create sparse value when pull");
DEFINE_bool(pslib_enable_create_feasign_randomly, false, "create feasign randomly");

namespace paddle {
namespace ps {

int32_t DownpourSparseTable::initialize() { 
    auto& profiler = CostProfiler::instance();
    _shards_task_pool.resize(24);//shard处理24并发
    for (int i = 0; i < _shards_task_pool.size(); ++i) {
        _shards_task_pool[i].reset(new ThreadPool<int>(1));
    }
    profiler.register_profiler("pslib_downpour_sparse_select_all");
    profiler.register_profiler("pslib_downpour_sparse_select_accessor");
    profiler.register_profiler("pslib_downpour_sparse_update_all");
    profiler.register_profiler("pslib_downpour_sparse_update_accessor");
    profiler.register_profiler("pslib_downpour_sparse_create_all");
    profiler.register_profiler("pslib_downpour_sparse_create_accessor");
    return 0;
}

int32_t DownpourSparseTable::initialize_shard() {
    _sparse_table_shard_num = FLAGS_pslib_sparse_table_shard_num;
    _avg_local_shard_num = SparseTable::sparse_local_shard_num(_sparse_table_shard_num, _shard_num);
    _real_local_shard_num = _avg_local_shard_num;
    if (_real_local_shard_num * (_shard_idx + 1) > _sparse_table_shard_num) {
        _real_local_shard_num = _sparse_table_shard_num - _real_local_shard_num * _shard_idx;
        _real_local_shard_num = _real_local_shard_num < 0 ? 0 : _real_local_shard_num;
    }
    _local_shards.reset(new shard_type[_real_local_shard_num]);
    return 0;
}

int32_t DownpourSparseTable::pull_sparse(float* pull_values, const uint64_t* keys, size_t num) {
    CostTimer timer("pslib_downpour_sparse_select_all");
    size_t value_size = _value_accesor->size() / sizeof(float);
    size_t mf_value_size = _value_accesor->mf_size() / sizeof(float);
    size_t select_value_size = _value_accesor->select_size() / sizeof(float);
    { //从table取值 or create    
        std::vector<std::future<int>> tasks(_real_local_shard_num);
        std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(_real_local_shard_num);
        for (size_t i = 0; i < num; ++i) {
            int shard_id = (keys[i] % _sparse_table_shard_num) % _avg_local_shard_num;
            task_keys[shard_id].push_back({keys[i], i});
        }
        
        std::atomic<uint32_t> missed_keys{0};
        for (size_t shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
            tasks[shard_id] = _shards_task_pool[shard_id % _shards_task_pool.size()]->AddTask(
            [this, shard_id, &task_keys, value_size, mf_value_size, select_value_size, 
            pull_values, keys, &missed_keys]() -> int {
                auto& keys = task_keys[shard_id];
                auto& local_shard = _local_shards[shard_id];
                float data_buffer[value_size];
                float* data_buffer_ptr = data_buffer;
                for (int i = 0; i < keys.size(); ++i) {
                    uint64_t key = keys[i].first;
                    auto itr = local_shard.find(key);
                    size_t data_size = value_size - mf_value_size;
                    if (itr == local_shard.end()) {
                        ++missed_keys;
                        if (FLAGS_pslib_create_value_when_push) {
                            memset(data_buffer, 0, sizeof(float) * data_size);
                        } else {
                            auto& feature_value = local_shard[key];
                            feature_value.resize(data_size);
                            float* data_ptr = const_cast<float*>(feature_value.data());
                            _value_accesor->create(&data_buffer_ptr, 1);
                            memcpy(data_ptr, data_buffer_ptr, data_size * sizeof(float));
                        }
                    } else {
                        data_size = itr.value().size();
                        memcpy(data_buffer_ptr, itr.value().data(), data_size * sizeof(float));
                    }
                    for (int mf_idx = data_size; mf_idx < value_size; ++mf_idx) {
                        data_buffer[mf_idx] = 0.0;
                    }
                    int pull_data_idx = keys[i].second;
                    float* select_data = pull_values + pull_data_idx * select_value_size;
                    _value_accesor->select(&select_data, (const float**)&data_buffer_ptr, 1);
                }
                return 0;
            }); 
        }
        for (size_t i = 0; i < _real_local_shard_num; ++i) {
            tasks[i].wait();
        }
        if (FLAGS_pslib_print_missed_key_num_every_push) {
            LOG(WARNING) << "total pull keys:" << num << " missed_keys:" << missed_keys.load();
        }
    }

    return 0;
}
        
int32_t DownpourSparseTable::push_sparse(
    const uint64_t* keys, const float* values, size_t num) {
    CostTimer timer("pslib_downpour_sparse_update_all");
    //构造value push_value的数据指针
    size_t value_col = _value_accesor->size() / sizeof(float);
    size_t mf_value_col = _value_accesor->mf_size() / sizeof(float);
    size_t update_value_col = _value_accesor->update_size() / sizeof(float);
    {
        std::vector<std::future<int>> tasks(_real_local_shard_num);
        std::vector<std::vector<std::pair<uint64_t, int>>> task_keys(_real_local_shard_num);
        for (size_t i = 0; i < num; ++i) {
            int shard_id = (keys[i] % _sparse_table_shard_num) % _avg_local_shard_num;
            task_keys[shard_id].push_back({keys[i], i});
        }
        for (size_t shard_id = 0; shard_id < _real_local_shard_num; ++shard_id) {
            tasks[shard_id] = _shards_task_pool[shard_id % _shards_task_pool.size()]->AddTask(
                [this, shard_id, value_col, mf_value_col, update_value_col, values, &task_keys]() -> int {
                auto& keys = task_keys[shard_id];
                auto& local_shard = _local_shards[shard_id];
                float data_buffer[value_col];
                float* data_buffer_ptr = data_buffer; 
                for (int i = 0; i < keys.size(); ++i) {
                    uint64_t key = keys[i].first;
                    uint64_t push_data_idx = keys[i].second;
                    const float* update_data = values + push_data_idx * update_value_col;
                    auto itr = local_shard.find(key);
                    if (itr == local_shard.end()) {
                        if (FLAGS_pslib_enable_create_feasign_randomly
                            && !_value_accesor->create_value(1, update_data)) {
                            continue;
                        }
                        auto value_size = value_col - mf_value_col;
                        auto& feature_value = local_shard[key];
                        feature_value.resize(value_size);
                        _value_accesor->create(&data_buffer_ptr, 1);
                        memcpy(const_cast<float*>(feature_value.data()), data_buffer_ptr, value_size * sizeof(float));
                        itr = local_shard.find(key);
                    }
                    auto& feature_value = itr.value();
                    float* value_data = const_cast<float*>(feature_value.data());
                    size_t value_size = feature_value.size();
                    if (value_size == value_col) { //已拓展到最大size, 则就地update
                        _value_accesor->update(&value_data, &update_data, 1);
                    } else {//拷入buffer区进行update，然后再回填，不需要的mf则回填时抛弃了
                        memcpy(data_buffer_ptr, value_data, value_size * sizeof(float));
                        _value_accesor->update(&data_buffer_ptr, &update_data, 1);
                        if (_value_accesor->need_extend_mf(data_buffer)) {
                            feature_value.resize(value_col);
                            value_data = const_cast<float*>(feature_value.data());
                            _value_accesor->create(&value_data, 1);
                        }
                        memcpy(value_data, data_buffer_ptr, value_size * sizeof(float));
                    }
                }
                return 0;
            });
        }
        for (size_t i = 0; i < _real_local_shard_num; ++i) {
            tasks[i].wait();
        }

    }
    /*
    //update && value 的转置
    thread_local Eigen::MatrixXf update_matrix;
    float* transposed_update_data[update_value_col];
    make_matrix_with_eigen(num, update_value_col, update_matrix, transposed_update_data);
    copy_array_to_eigen(values, update_matrix);

    thread_local Eigen::MatrixXf value_matrix;
    float* transposed_value_data[value_col];
    make_matrix_with_eigen(num, value_col, value_matrix, transposed_value_data);
    copy_matrix_to_eigen((const float**)(value_ptrs->data()), value_matrix);

    //批量update
    {
        CostTimer accessor_timer("pslib_downpour_sparse_update_accessor");
        _value_accesor->update(transposed_value_data, (const float**)transposed_update_data, num);
    }
    copy_eigen_to_matrix(value_matrix, value_ptrs->data());
    */
    return 0;
}

int32_t DownpourSparseTable::shrink() {
    //TODO implement with multi-thread
    for (size_t i = 0; i < _real_local_shard_num; ++i) {
        auto& shard = _local_shards[i];
        for (auto it = shard.begin(); it != shard.end();) {
            if (_value_accesor->shrink(it.value().data())) {
                it = shard.erase(it);
            } else {
                ++it;
            }
        }
    }
    return 0;
}
        
int32_t DownpourSparseTable::save(const std::string& path, const std::string& param) {
    int save_param = atoi(param.c_str());//batch_model:0  xbox:1
    size_t file_start_idx = _avg_local_shard_num * _shard_idx;
    std::string table_path = table_dir(path);
    _afs_client.remove(format_string("%s/part-%03d-*", table_path.c_str(), _shard_idx));
    int thread_num = _real_local_shard_num < 20 ? _real_local_shard_num : 20;

    //std::atomic<uint32_t> feasign_size;
    std::atomic<uint32_t> feasign_size_all;
    //feasign_size = 0;

    omp_set_num_threads(thread_num);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < _real_local_shard_num; ++i) {
        FsChannelConfig channel_config;
        if (_config.compress_in_save()) {
            channel_config.path = format_string("%s/part-%03d-%05d.gz", 
                table_path.c_str(), _shard_idx, file_start_idx + i);
        } else {
            channel_config.path = format_string("%s/part-%03d-%05d", 
                table_path.c_str(), _shard_idx, file_start_idx + i);
        }
        channel_config.converter = _value_accesor->converter(save_param).converter;
        channel_config.deconverter = _value_accesor->converter(save_param).deconverter;
        bool is_write_failed = false;
        int feasign_size = 0;
        do {
            feasign_size = 0;
            is_write_failed = false;
            auto write_channel = _afs_client.open_w(channel_config, 1024 * 1024 * 40);
            auto& shard = _local_shards[i];
            for (auto it = shard.begin(); it != shard.end(); ++it) {
                if (_value_accesor->save(it.value().data(), save_param)) {
                    std::string format_value = _value_accesor->
                        parse_to_string(it.value().data(), save_param);
                    if (0 != write_channel->write_line(
                                format_string("%lu %s", it.key(), format_value.c_str()))) {
                        LOG(ERROR) << "DownpourSparseTable save failed, "
                            "path:" << channel_config.path << ", retry it!";
                        is_write_failed = true;
                        break;
                    }
                    ++feasign_size;
                }
            }
            write_channel->close();
            if (is_write_failed) {
                _afs_client.remove(channel_config.path);
            }
        } while (is_write_failed);
        feasign_size_all += feasign_size;
    }
    LOG(INFO) << "DownpourSparseTable save success, path:" << 
        format_string("%s/%03d/part-%03d-", path.c_str(), _config.table_id(), _shard_idx)
        << " from " << file_start_idx << " to " << file_start_idx + _real_local_shard_num - 1;
    return feasign_size_all;
}

int32_t DownpourSparseTable::load(const std::string& path, const std::string& param) {
    std::string table_path = table_dir(path);
    auto file_list = _afs_client.list(table_path);
    if (file_list.size() !=  _sparse_table_shard_num) {
        LOG(WARNING) << "DownpourSparseTable file_size:" << file_list.size() 
            << " not equal to shard_num:" << _sparse_table_shard_num;
        //TODO load
        return -1;
    }
    if (file_list.size() == 0) {
        LOG(WARNING) << "DownpourSparseTable load file is empty, path:" << path;
        return -1;
    }

    size_t file_start_idx = _shard_idx * _avg_local_shard_num;
    return load(file_start_idx, file_start_idx + _real_local_shard_num, file_list, param);
}

//加载path目录下数据[start_idx, end_idx)
int32_t DownpourSparseTable::load(size_t start_idx, size_t end_idx,
        const std::vector<std::string>& file_list, const std::string& param) {
    if (start_idx >= file_list.size()) {
        return 0;
    }
    int load_param = atoi(param.c_str());
    size_t feature_value_size = _value_accesor->size() / sizeof(float);
    end_idx = end_idx < _sparse_table_shard_num ? end_idx : _sparse_table_shard_num;
    int thread_num = (end_idx - start_idx) < 15 ? (end_idx - start_idx) : 15;
    omp_set_num_threads(thread_num);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = start_idx; i < end_idx; ++i) {
        FsChannelConfig channel_config;
        channel_config.path = file_list[i];
        channel_config.converter = _value_accesor->converter(load_param).converter;
        channel_config.deconverter = _value_accesor->converter(load_param).deconverter;
      
        bool is_read_failed = false;
        do {
            is_read_failed = false;
            std::string line_data;
            auto read_channel = _afs_client.open_r(channel_config);
            char *end = NULL;
            auto& shard = _local_shards[i % _avg_local_shard_num];
            try {
                while (read_channel->read_line(line_data) == 0 && line_data.size() > 1) {
                    uint64_t key = std::strtoul(line_data.data(), &end, 10);
                    if (FLAGS_pslib_open_strict_check) {
                        if (key % _sparse_table_shard_num != i) {
                            LOG(WARNING) << "DownpourSparseTable key:" << key << " not match shard," 
                            << " file_idx:" << i
                            << " shard num:" << _sparse_table_shard_num
                            << " file:" << channel_config.path;
                            continue;
                        }
                    }
                    auto& value = shard[key];
                    value.resize(feature_value_size);
                    _value_accesor->parse_from_string(++end, value.data());
                }
            } catch(...) {
                is_read_failed = true;
                LOG(ERROR) << "DownpourSparseTable load failed:" << channel_config.path << ", retry it!";
            }
        } while (is_read_failed);
    }
    LOG(INFO) << "DownpourSparseTable load success, path from " 
        << file_list[start_idx] << " to " << file_list[end_idx - 1];
    return 0;
}


} //namespace ps
} //namespace paddle
