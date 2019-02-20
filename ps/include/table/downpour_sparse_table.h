#pragma once
#include <assert.h>
#include <string>
#include <omp.h>
#include "accessor.h"
#include "table.h"
#include "mct/hash-map.hpp"
#include "common/chunk_allocator.h"
#include "communicate/ps_env.h"
#include "common/thread_pool.h"
#include <pthread.h>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/repetition/enum.hpp>

namespace paddle {
namespace ps {
    static const int DOWNPOUR_SPARSE_SHARD_BUCKET_NUM_BITS = 6;
    static const size_t DOWNPOUR_SPARSE_SHARD_BUCKET_NUM = (size_t)1 << DOWNPOUR_SPARSE_SHARD_BUCKET_NUM_BITS;
    template<class KEY, class VALUE>
    struct alignas(64) DownpourSparseTableShard {
        public:
            typedef typename mct::closed_hash_map<KEY, mct::Pointer, std::hash<KEY>> map_type;
            struct iterator {
                typename map_type::iterator it;
                size_t bucket;
                map_type* buckets;
                friend bool operator==(const iterator& a, const iterator& b) {
                    return a.it == b.it;
                }
                friend bool operator!=(const iterator& a, const iterator& b) {
                    return a.it != b.it;
                }
                const KEY& key() const {
                    return it->first;
                }
                VALUE& value() const {
                    return *(VALUE*)(void*)it->second;
                }
                iterator& operator++() {
                    ++it;

                    while (it == buckets[bucket].end() && bucket + 1 < DOWNPOUR_SPARSE_SHARD_BUCKET_NUM) {
                        it = buckets[++bucket].begin();
                    }

                    return *this;
                }
                iterator operator++(int) {
                    iterator ret = *this;
                    ++*this;
                    return ret;
                }
            };
            struct local_iterator {
                typename map_type::iterator it;
                friend bool operator==(const local_iterator& a, const local_iterator& b) {
                    return a.it == b.it;
                }
                friend bool operator!=(const local_iterator& a, const local_iterator& b) {
                    return a.it != b.it;
                }
                const KEY& key() const {
                    return it->first;
                }
                VALUE& value() const {
                    return *(VALUE*)(void*)it->second;
                }
                local_iterator& operator++() {
                    ++it;
                    return *this;
                }
                local_iterator operator++(int) {
                    return {it++};
                }
            };

            ~DownpourSparseTableShard() {
                clear();
            }
            bool empty() {
                return _alloc.size() == 0;
            }
            size_t size() {
                return _alloc.size();
            }
            void set_max_load_factor(float x) {
                for (size_t bucket = 0; bucket < DOWNPOUR_SPARSE_SHARD_BUCKET_NUM; bucket++) {
                    _buckets[bucket].max_load_factor(x);
                }
            }
            size_t bucket_count() {
                return DOWNPOUR_SPARSE_SHARD_BUCKET_NUM;
            }
            size_t bucket_size(size_t bucket) {
                return _buckets[bucket].size();
            }
            void clear() {
                for (size_t bucket = 0; bucket < DOWNPOUR_SPARSE_SHARD_BUCKET_NUM; bucket++) {
                    map_type& data = _buckets[bucket];
                    for (auto it = data.begin(); it != data.end(); ++it) {
                        _alloc.release((VALUE*)(void*)it->second);
                    }
                    data.clear();
                }
            }
            iterator begin() {
                auto it = _buckets[0].begin();
                size_t bucket = 0;
                while (it == _buckets[bucket].end() && bucket + 1 < DOWNPOUR_SPARSE_SHARD_BUCKET_NUM) {
                    it = _buckets[++bucket].begin();
                }
                return {it, bucket, _buckets};
            }
            iterator end() {
                return {_buckets[DOWNPOUR_SPARSE_SHARD_BUCKET_NUM - 1].end(), DOWNPOUR_SPARSE_SHARD_BUCKET_NUM - 1, _buckets};
            }
            local_iterator begin(size_t bucket) {
                return {_buckets[bucket].begin()};
            }
            local_iterator end(size_t bucket) {
                return {_buckets[bucket].end()};
            }
            iterator find(const KEY& key) {
                size_t hash = _hasher(key);
                size_t bucket = compute_bucket(hash);
                auto it = _buckets[bucket].find_with_hash(key, hash);
                if (it == _buckets[bucket].end()) {
                    return end();
                }
                return {it, bucket, _buckets};
            }
            VALUE& operator[](const KEY& key) {
                return emplace(key).first.value();
            }
            std::pair<iterator, bool> insert(const KEY& key, const VALUE& val) {
                return emplace(key, val);
            }
            std::pair<iterator, bool> insert(const KEY& key, VALUE && val) {
                return emplace(key, std::move(val));
            }
            template<class... ARGS>
            std::pair<iterator, bool> emplace(const KEY& key, ARGS && ... args) {
                size_t hash = _hasher(key);
                size_t bucket = compute_bucket(hash);
                auto res = _buckets[bucket].insert_with_hash( {key, NULL}, hash);

                if (res.second) {
                    res.first->second = _alloc.acquire(std::forward<ARGS>(args)...);
                }

                return {{res.first, bucket, _buckets}, res.second};
            }
            iterator erase(iterator it) {
                _alloc.release((VALUE*)(void*)it.it->second);
                size_t bucket = it.bucket;
                auto it2 = _buckets[bucket].erase(it.it);
                while (it2 == _buckets[bucket].end() && bucket + 1 < DOWNPOUR_SPARSE_SHARD_BUCKET_NUM) {
                    it2 = _buckets[++bucket].begin();
                }
                return {it2, bucket, _buckets};
            }
            void quick_erase(iterator it) {
                _alloc.release((VALUE*)(void*)it.it->second);
                _buckets[it.bucket].quick_erase(it.it);
            }
            local_iterator erase(size_t bucket, local_iterator it) {
                _alloc.release((VALUE*)(void*)it.it->second);
                return {_buckets[bucket].erase(it.it)};
            }
            void quick_erase(size_t bucket, local_iterator it) {
                _alloc.release((VALUE*)(void*)it.it->second);
                _buckets[bucket].quick_erase(it.it);
            }
            size_t erase(const KEY& key) {
                auto it = find(key);
                if (it == end()) {
                    return 0;
                }
                quick_erase(it);
                return 1;
            }
            size_t compute_bucket(size_t hash) {
                if (DOWNPOUR_SPARSE_SHARD_BUCKET_NUM == 1) {
                    return 0;
                } else {
                    return hash >> (sizeof(size_t) * 8 - DOWNPOUR_SPARSE_SHARD_BUCKET_NUM_BITS);
                }
            }
        private:
            map_type _buckets[DOWNPOUR_SPARSE_SHARD_BUCKET_NUM];
            ChunkAllocator<VALUE> _alloc;
            std::hash<KEY> _hasher;
    };

    class DownpourFixedFeatureValue {
    public:
        DownpourFixedFeatureValue() {}
        ~DownpourFixedFeatureValue() {}
        float* data() {
            return _data.data();
        }
        size_t size() {
            return _data.size();
        }
        void resize(size_t size) {
            _data.resize(size);
        }
    private:
        std::vector<float> _data;
    };

    class DownpourSparseTable : public SparseTable {
        public:
            typedef DownpourSparseTableShard<uint64_t, DownpourFixedFeatureValue> shard_type;
            DownpourSparseTable() {}
            virtual ~DownpourSparseTable() {}            

            virtual int32_t initialize() override;
            virtual int32_t initialize_shard() override;
            
            
            virtual int32_t pull_sparse(float* pull_values, const uint64_t* keys, size_t num);
            virtual  PooledData pull_sparse(const uint64_t* keys, size_t num) {
                auto pull_data = _data_pool.get();    
                pull_data->resize((_value_accesor->select_size() / sizeof(float)) * num);
                pull_sparse(pull_data->data(), keys, num); 
                return pull_data;
            }
            virtual int32_t push_sparse(const uint64_t* keys, const float* values, size_t num);
            
            virtual int32_t flush() override {
                return 0;
            }
            virtual int32_t shrink() override;
            virtual void clear() override {
                for (size_t i = 0; i < _real_local_shard_num; ++i) {
                    _local_shards[i].clear();
                }
            }

            virtual int32_t save(const std::string& path, const std::string& param) override;
            //加载path目录下数据
            virtual int32_t load(const std::string& path, const std::string& param) override;
            //加载path目录下数据[start_idx, end_idx)
            int32_t load(size_t start_idx, size_t end_idx,
                    const std::vector<std::string>& file_list, const std::string& param);

        private:
            int _avg_local_shard_num;                  //平均每个table下local_shard个数
            int _real_local_shard_num;                 //该table下实际local_shard个数(尾数问题)
            int _sparse_table_shard_num;
            std::unique_ptr<shard_type[]> _local_shards;
            std::vector<std::shared_ptr<ThreadPool<int>>> _shards_task_pool;
    };
    
    
} //namespace ps
} //namespace paddle
