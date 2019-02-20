#pragma once
#include <assert.h>
#include <string>
#include <pthread.h>
#include "table.h"
#include "accessor.h"
#include "Eigen/Dense"
#include "common/thread_pool.h"

namespace paddle {
namespace ps {

class DownpourDenseTable : public DenseTable {
public:
    explicit DownpourDenseTable() {}
    virtual ~DownpourDenseTable() {} 
    virtual int32_t initialize() override;
    virtual int32_t initialize_shard() override {
        return 0;
    }
    
    virtual int32_t create_dense(size_t num, bool init);
    
    virtual  PooledData pull_dense(size_t num) override {
        auto pull_data = _data_pool.get();
        pull_data->resize(num * _value_accesor->select_size() / sizeof(float));
        pull_dense(pull_data->data(), num);
        return pull_data;
    }
    virtual int32_t pull_dense(float* pull_values, size_t num) override;
    virtual int32_t push_dense_param(const float* values, size_t num) override;
    virtual int32_t push_dense(const float* values, size_t num) override;

    int32_t load(const std::string& path, const std::string& param) override;
    int32_t save(const std::string& path, const std::string& param) override;
    
    virtual int32_t flush() override {
        return 0;
    }
    virtual int32_t shrink() override {
        return 0;
    }
    virtual void clear() override {
        return;
    }
private:
    int32_t dense_update(const float* update_values, size_t num, size_t start_idx, size_t update_num);

    std::mutex _mutex;
    Eigen::MatrixXf _data;
    std::vector<std::shared_ptr<ThreadPool<int>>> _shards_task_pool;
};

} //namespace ps
} //namespace paddle
