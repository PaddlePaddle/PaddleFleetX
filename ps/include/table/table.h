#pragma once
#include <assert.h>
#include <string>
#include <atomic>
#include "accessor.h"
#include "proto/ps.pb.h"
#include "common/timer.h"
#include "common/pool.h"

DECLARE_bool(pslib_is_debug);
DECLARE_bool(pslib_is_debug_push_sparse);
DECLARE_int32(pslib_sparse_table_shard_num);
DECLARE_bool(pslib_is_update_grident_thread_save);

namespace paddle {
namespace ps {
    typedef ObjectPool<std::vector<float>> DataPool;
    typedef ObjectPool<std::vector<float*>> DataPtrPool;
    typedef ObjectPool<std::vector<float>>::PooledObject PooledData;
    
    class Table {
        public:
            explicit Table() {}
            virtual ~Table() {}
            virtual int32_t initialize(const TableParameter& config,
                const FsClientParameter& fs_config) final;
            
            virtual  int32_t pull_dense(float* values, size_t num) = 0;
            virtual  PooledData pull_dense(size_t num) = 0;
            virtual int32_t push_dense(const float* values, size_t num) = 0;
            virtual int32_t push_dense_param(const float* values, size_t num) {return 0;}
            virtual int32_t pull_sparse(float* values, const uint64_t* keys, size_t num) = 0;
            virtual PooledData pull_sparse(const uint64_t* keys, size_t num) = 0;
            virtual int32_t push_sparse(const uint64_t* keys, const float* values, size_t num) = 0;

            virtual void clear() = 0;
            virtual int32_t flush() = 0;
            virtual int32_t shrink() = 0;
            //指定加载路径
            virtual int32_t load(const std::string& path, const std::string& converter) = 0;
            //指定保存路径
            virtual int32_t save(const std::string& path, const std::string& converter) = 0;
            virtual int32_t set_shard(size_t shard_idx, size_t shard_num) final {
                _shard_idx = shard_idx;
                _shard_num = shard_num;
                return initialize_shard();
            }
            
            inline std::shared_ptr<ValueAccessor> value_accesor() {
                return _value_accesor;
            }

        protected:
            virtual int32_t initialize() = 0;
            virtual int32_t initialize_accessor() final;
            virtual int32_t initialize_shard() = 0;
            virtual std::string table_dir(const std::string& model_dir) {
                return format_string("%s/%03d/", model_dir.c_str(), _config.table_id());
            }
            
            size_t _shard_idx;           // table 分片编号
            size_t _shard_num;           // table 分片总数
            AfsClient _afs_client;
            TableParameter _config;
            DataPool _data_pool; 
            DataPtrPool _data_ptr_pool; 
            std::shared_ptr<ValueAccessor> _value_accesor;
    };
    REGISTER_REGISTERER(Table);

    class SparseTable : public Table {
        public:
            explicit SparseTable() {}
            virtual ~SparseTable() {}            
            
            virtual  int32_t pull_dense(float* values, size_t num) override {
                assert(0);
                return 0;
            }
            virtual int32_t push_dense(const float* values, size_t num) override {
                assert(0);
                return 0;
            }
            virtual PooledData pull_dense(size_t num) override {
                assert(0);
                PooledData x;
                return x;
            }
            static int32_t sparse_local_shard_num(uint32_t shard_num, uint32_t server_num) {
                if (shard_num % server_num == 0) {
                    return shard_num / server_num;
                }
                size_t local_shard_num = shard_num / server_num + 1;
                return local_shard_num;
            }

            static size_t get_sparse_shard(uint32_t shard_num, uint32_t server_num, uint64_t key) {
                return (key % shard_num) / sparse_local_shard_num(shard_num, server_num);
            }
    };


    class DenseTable : public Table {
        public:
            explicit DenseTable() {}
            virtual ~DenseTable() {}
            virtual int32_t pull_sparse(float* values, const uint64_t* keys, size_t num) override {
                assert(0);
                return 0;
            }
            virtual int32_t push_sparse(const uint64_t* keys, const float* values, size_t num) override {
                assert(0);
                return 0;
            }
            virtual PooledData pull_sparse(const uint64_t* keys, size_t num) override {
                assert(0);
                PooledData x;
                return x;
            }
            virtual int32_t push_dense_param(const float* values, size_t num) override {
                assert(0);
                return 0;
            }
            virtual int32_t shrink() override {
                return 0;
            }
    };

    class TableManager {
    public:
        static TableManager& instance() {
            static TableManager manager;
            return manager;
        }
        int32_t initialize();
    private:
        TableManager() {}
        ~TableManager() {}
    };
    
}
}
