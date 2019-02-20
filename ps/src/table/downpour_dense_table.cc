#include "table/downpour_dense_table.h"
#include "common/matrix.h"

DEFINE_int32(pslib_server_dense_update_thread_num, 10, "dense_update_thread_num in server");

namespace paddle {
namespace ps {
int32_t DownpourDenseTable::initialize() {
    auto& profiler = CostProfiler::instance();
    _shards_task_pool.resize(FLAGS_pslib_server_dense_update_thread_num);
    for (int i = 0; i < _shards_task_pool.size(); ++i) {
        _shards_task_pool[i].reset(new ThreadPool<int>(1));
    }
    profiler.register_profiler("pslib_downpour_dense_select_all");
    profiler.register_profiler("pslib_downpour_dense_select_accessor");
    profiler.register_profiler("pslib_downpour_dense_update_all");
    profiler.register_profiler("pslib_downpour_dense_update_accessor");
    profiler.register_profiler("pslib_downpour_dense_create_all");
    profiler.register_profiler("pslib_downpour_dense_create_accessor");
    return 0;
}

int32_t DownpourDenseTable::create_dense(size_t num, bool init) {
    if (_data.rows() != num) {
        CostTimer create_timer("pslib_downpour_dense_create_all");
        std::lock_guard<std::mutex> lock(_mutex);
        if (_data.rows() != num) {
            _data.setZero(num, _value_accesor->size() / sizeof(float));
            float*  create_ptrs[_data.cols()];
            for (size_t i = 0; i < _data.cols(); ++i) {
                create_ptrs[i] = _data.data() + i * num;
            }
            if (init) {
                CostTimer accessor_timer("pslib_downpour_dense_create_accessor");
                _value_accesor->create(create_ptrs, num);
            }
        }
    }
    return 0; 
}
        
int32_t DownpourDenseTable::pull_dense(float* pull_values, size_t num) {
    CostTimer timer("pslib_downpour_dense_select_all");
    create_dense(num, true); 
    //构造value pull_value create_value的数据指针
    float*  value_ptrs[_data.cols()];
    for (size_t i = 0; i < _data.cols(); ++i) {
        value_ptrs[i] = _data.data() + i * num;
    }
    
    thread_local Eigen::MatrixXf select_result;
    size_t select_data_col = _value_accesor->select_size() / sizeof(float);
    float*  pull_value_ptrs[select_data_col];
    make_matrix_with_eigen(num, select_data_col, select_result, pull_value_ptrs);
    
    //批量select到pull_value
    {
        CostTimer accessor_timer("pslib_downpour_dense_select_accessor");
        _value_accesor->select(pull_value_ptrs, (const float**)value_ptrs, num);
    }
    copy_eigen_to_array(select_result, pull_values);
    if (FLAGS_pslib_is_debug) {
        LOG(INFO) << "[label:pull_dense]\nvalue:\n" << matrix_to_string(_data)
            << "\nselect_value:\n" << matrix_to_string(select_result);
    }
    return 0;
}

int32_t DownpourDenseTable::push_dense_param(const float* values, size_t num) {
    create_dense(num, true);
    
    thread_local Eigen::MatrixXf update_value;
    size_t update_data_col = _value_accesor->update_size() / sizeof(float);
    float* transposed_push_data[update_data_col];
    make_matrix_with_eigen(num, update_data_col, update_value, transposed_push_data);
    copy_array_to_eigen(values, update_value);
    
    float*  value_ptrs[_data.cols()];
    for (size_t i = 0; i < _data.cols(); ++i) {
        value_ptrs[i] = _data.data() + i * num;
    }

    //批量update
    std::lock_guard<std::mutex> lock(_mutex);
    _value_accesor->set_weight(value_ptrs, (const float**)transposed_push_data, num);
    return 0;
}

int32_t DownpourDenseTable::dense_update(const float* update_values,
    size_t num, size_t start_idx, size_t update_num) {
    size_t update_data_col = _value_accesor->update_dim();
    const float* push_ptrs[update_data_col];
    for (size_t i = 0; i < update_data_col; ++i) {
        push_ptrs[i] = update_values + i * num + start_idx;
    }
    float*  value_ptrs[_data.cols()];
    for (size_t i = 0; i < _data.cols(); ++i) {
        value_ptrs[i] = _data.data() + i * num + start_idx;
    }
    _value_accesor->update(value_ptrs, push_ptrs, update_num);
    return 0;
}

int32_t DownpourDenseTable::push_dense(const float* values, size_t num) {
    CostTimer timer("pslib_downpour_dense_update_all");
    create_dense(num, false);
    //批量update
    {
        if (FLAGS_pslib_is_update_grident_thread_save) {
            CostTimer timer("pslib_downpour_dense_update_accessor");
            size_t avg_num = num / FLAGS_pslib_server_dense_update_thread_num + 1;
            std::vector<std::future<int>> tasks(FLAGS_pslib_server_dense_update_thread_num);
            for (size_t shard_id = 0; shard_id < 10; ++shard_id) {
                tasks[shard_id] = _shards_task_pool[shard_id]->AddTask(
                    [this, shard_id, &values, num, avg_num]() -> int {
                        size_t start_idx = avg_num * shard_id;
                        if (start_idx >= num) {
                            return 0;
                        }
                        size_t update_num = avg_num;
                        if (start_idx + avg_num > num) {
                            update_num = num - start_idx;
                        }
                        dense_update(values, num, start_idx, update_num);
                        return 0;
                });
            }
            for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
                tasks[shard_id].wait();
            }
        } else { //无锁更新则无需多线程
            dense_update(values, num, 0, num);
        }
    }
    return 0;
}

int32_t DownpourDenseTable::load(const std::string& path, const std::string& param) {
    std::string table_path = table_dir(path);
    auto file_list = _afs_client.list(table_path);
    size_t dim_num_per_file = _value_accesor->fea_dim() / file_list.size() + 1;
    size_t dim_num_per_shard = _value_accesor->fea_dim() / _shard_num + 1;
    size_t start_dim_idx = dim_num_per_shard * _shard_idx; 
    size_t start_file_idx = start_dim_idx / dim_num_per_file;
    size_t end_file_idx = (dim_num_per_shard * (_shard_idx + 1)) / dim_num_per_file;
    end_file_idx = end_file_idx < file_list.size() ? end_file_idx : file_list.size() - 1;
    
    size_t dim_col = _value_accesor->size() / sizeof(float);
    _data.resize(dim_num_per_shard, dim_col);
    
    int load_param = atoi(param.c_str());
    FsChannelConfig channel_config;

    channel_config.converter = _value_accesor->converter(load_param).converter;
    channel_config.deconverter = _value_accesor->converter(load_param).deconverter;
    bool is_read_failed = false;
    do {
        is_read_failed = false;
        try {
            size_t dim_idx = 0;
            float dim_data_buffer[dim_col];
            std::string line_data;
            for (int i = start_file_idx; i < end_file_idx + 1; ++i) {
                channel_config.path = file_list[i]; 
                auto read_channel = _afs_client.open_r(channel_config);
                size_t file_start_idx = start_dim_idx - i * dim_num_per_file;
                for (size_t file_dim_idx = 0; file_dim_idx < dim_num_per_file; ++file_dim_idx) {
                    if (read_channel->read_line(line_data) != 0) {
                        break;
                    }
                    if (dim_idx >= dim_num_per_shard) {
                        break;
                    }
                    if (file_dim_idx < file_start_idx) {
                        continue;
                    }
                    _value_accesor->parse_from_string(line_data, dim_data_buffer);
                    for (size_t col_idx = 0; col_idx < dim_col; ++col_idx) {
                        _data(dim_idx, col_idx) = dim_data_buffer[col_idx];
                    }
                    ++dim_idx;
                }
                start_dim_idx += dim_num_per_file - file_start_idx;
                LOG(INFO) << "DownpourDenseTable load success, path:" << channel_config.path;
            }
        } catch (...) { 
            is_read_failed = true;
            LOG(ERROR) << "DownpourDenseTable load failed, retry it! path:" << channel_config.path;
        }
    } while (is_read_failed);
    return 0;
}

int32_t DownpourDenseTable::save(const std::string& path, const std::string& param) {
    int save_param = atoi(param.c_str());
    uint32_t feasign_size;

    FsChannelConfig channel_config;
    if (_config.compress_in_save()) {
        channel_config.path = format_string("%s/part-%03d.gz", 
            table_dir(path).c_str(), _shard_idx);
    } else {
        channel_config.path = format_string("%s/part-%03d", 
            table_dir(path).c_str(), _shard_idx);
    }
    _afs_client.remove(channel_config.path);
    channel_config.converter = _value_accesor->converter(save_param).converter;
    channel_config.deconverter = _value_accesor->converter(save_param).deconverter;

    float dim_data_buffer[_data.cols()];
    bool is_write_failed = false;
    std::vector<std::string> result_buffer;
    result_buffer.reserve(_data.rows());
    for (auto i = 0u; i < _data.rows(); ++i) {
        for (int j = 0; j < _data.cols(); ++j) {
            dim_data_buffer[j] = _data(i, j);
        }
        std::string item_data = _value_accesor->
            parse_to_string(dim_data_buffer, save_param);
        result_buffer.emplace_back(std::move(item_data));
    }
    do {
        is_write_failed = false;
        feasign_size = 0;
        // 40M
        auto write_channel = _afs_client.open_w(channel_config, 1024 * 1024 * 40);
        for (auto& t : result_buffer) {
            if (0 != write_channel->write_line(t)) {
                LOG(ERROR) << "DownpourDenseTable save failed, "
                    "path:" << channel_config.path << ", retry it!";
                is_write_failed = true;
                break;
            }
        }
        ++feasign_size;
        write_channel->close();
        if (is_write_failed) {
            _afs_client.remove(channel_config.path);
        }
    } while (is_write_failed);
    LOG(INFO) << "DownpourDenseTable save success, path:" << channel_config.path;
    return feasign_size;
}

} //namespace ps
} //namespace paddle
