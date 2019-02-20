#ifndef BAIDU_FEED_MLARCH_BABYLON_CACHE_H
#define BAIDU_FEED_MLARCH_BABYLON_CACHE_H

#include <inttypes.h>
#include <sched.h>
#include <stdlib.h>
#include <type_traits>
#include <random>
#include <atomic>
#include <thread>
#include <future>
#include <chrono>
#include <tbb/concurrent_unordered_map.h>
#include <base/logging.h>
#include <base/time.h>
#include <baidu/feed/mlarch/babylon/expect.h>

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

// 非class enum可以支持默认转整数
// 包一层同名namespace用来做隔离
#define baidu_feed_mlarch_babylon_cache_transient_cache_option TransientCacheOption
namespace baidu_feed_mlarch_babylon_cache_transient_cache_option {
    enum TransientCacheOption {
        // 增加引用计数来提供安全析构能力
        // 否则需要用户确保load callback不会和析构并发执行
        // 例如全部排空后析构
        // 一般cache都是常驻，服务自身优雅关闭后再析构即可
        SAFE_DESTRUCTION = 0x1,

        // 提高原子操作等级来提供安全cache淘汰
        // 如果cache duration时间足够gc即（gctime << cache duration * amplification * 2）
        // 可以不用开启
        SAFE_EXPIRATION = 0x2,

        // 当超过过期时间后，不再等待已有的visitor结束，强制释放缓存内容
        // 在load或者done可能有潜在死锁风险的时候，可以强制跳出
        // 但由于visitor尚未释放，所以有潜在crash风险，慎用
        FORCE_EXPIRATION = 0x4,

        // 默认都不开启
        DEFAULT = 0x0,
    };
};
#undef baidu_feed_mlarch_babylon_cache_transient_cache_option

template <typename K, typename V, typename H = ::tbb::tbb_hash<K>, uint32_t O = TransientCacheOption::DEFAULT>
class TransientCache {
private:
    class Slice;
    class Node;

public:
    // 提供给用户load function使用，用于接收load到的结果
    // 记录了发起时的slice epoch，解决应对ABA
    class LoadCallback {
    public:
        // load成功，将value传入
        inline void done(::std::shared_ptr<const V>&& value) noexcept;

        // load失败，将error code传入
        inline void done(int32_t error_code) noexcept;

    private:
        Slice& _slice;
        Node& _node;
        int64_t _epoch;

        LoadCallback(Slice& slice, Node& node, int64_t epoch) noexcept :
            _slice(slice), _node(node), _epoch(epoch) {
            // 安全析构模式下，创建LoadCallback也会计数，用来跟踪未完成的load
            if (unlikely(O & TransientCacheOption::SAFE_DESTRUCTION)) {
                _slice.enter();
            }
        }

        friend class Slice;
    };

    enum ReturnCode {
        ERROR = -1,
        OK = 0,
        BUSY = 1,
        EMPTY = 2
    };

    // 用户提供的load function，pioneer会调用来取值
    typedef ::std::function<int32_t(const K&, LoadCallback)> LoadFunction;

    // 用户提供的value callback function，ready之后会调用来通知用户
    typedef ::std::function<void(int32_t,
        const ::std::shared_ptr<const V>&)> ValueCallbackFunction;

    // 用户提供的value callback task，ready之后会调用来通知用户
    // 对move only的value callback提供支持
    // 对同步调用提供支持
    typedef ::std::packaged_task<void(int32_t,
        const ::std::shared_ptr<const V>&)> ValueCallbackTask;

    // 构造函数，RAII模式，会后台启动一个GC线程
    TransientCache(const LoadFunction& load_function, int64_t cache_duration_us,
        uint32_t amplification = 10, double jitter = 0.0,
        const ::std::string& name = "") noexcept;

    // 析构函数，停止后台的GC线程
    ~TransientCache() {
        stop_gc();

        if (unlikely(O & TransientCacheOption::SAFE_DESTRUCTION)) {
            bool all_finish = false;
            while (!all_finish) {
                all_finish = true;
                for (auto& slice : _slices) {
                    if (OK != slice.reset_to_epoch(INT64_MIN)) {
                        all_finish = false;
                    }
                }
                if (!all_finish) {
                    usleep(100000);
                }
            }
        }
    }

    // ==================================================================
    // 异步取值接口
    // 在cache duration内同一个key的访问者只由pioneer发起load function
    // ready后激活包括自己在内所有访问者的callback
    // load function运行在pioneer的线程中
    // callback运行在调用LoadCallback的done线程中（值尚未ready）
    // 或者调用get的线程当中（值已经ready）
    // 是否使用/使用何种异步机制由用户在load和callback中实现
    // 参数
    // key:               需要查询的key
    // callback:          传入callback，ready后通过callback向用户通知结果
    //                    callback在调用后即不可再使用
    //                    如果因为失败等原因需要再次发起重试，需要重新构造
    //                    或者使用canceled_callback机制
    //                    要求可以move构造
    // canceled_callback: 当直接使用lambda等非function/task的callback时
    //                    可以额外传入一个function/task，返回失败未处理时
    //                    会将callback放入canceled_callback中，此时用户可以
    //                    选择直接调用，或者用来发起退避重试，避免再次构造
    // 返回值
    // OK:      表示正常进入处理流程，callback会在ready后被调用
    // BUSY:    仅在GC延迟时出现，未进入处理流程，可以退避重试

    // 自身不是，但是可以构造成callback function的版本
    template <typename C, typename std::enable_if<
        !::std::is_same<ValueCallbackFunction, C>::value
        && ::std::is_copy_constructible<C>::value
        && ::std::is_move_constructible<C>::value, int>::type = 0>
    int32_t get(const K& key, C&& callback,
        ValueCallbackFunction* canceled_callback = nullptr) noexcept {
        ValueCallbackFunction function = ::std::move(callback);
        auto ret = get(key, ::std::move(function));
        if (unlikely(ret != 0 && canceled_callback != nullptr)) {
            *canceled_callback = ::std::move(function);
        }
        return ret;
    }

    // 自身不是，但是只能构造成callback task的版本
    template <typename C, typename std::enable_if<
        !::std::is_same<ValueCallbackTask, C>::value
        && !::std::is_copy_constructible<C>::value
        && ::std::is_move_constructible<C>::value, int>::type = 0>
    int32_t get(const K& key, C&& callback,
        ValueCallbackTask* canceled_callback = nullptr) noexcept {
        ValueCallbackTask task(::std::move(callback));
        auto ret = get(key, ::std::move(task));
        if (unlikely(ret != 0 && canceled_callback != nullptr)) {
            *canceled_callback = ::std::move(task);
        }
        return ret;
    }

    // 最终统一通过callback function和callback task交互
    template <typename C, typename ::std::enable_if<
        ::std::is_same<ValueCallbackFunction, C>::value
        || ::std::is_same<ValueCallbackTask, C>::value, int>::type = 0>
    int32_t get(const K& key, C&& callback) noexcept {
        return get_with_timestamp(key, ::std::move(callback));
    }
    // 异步取值接口
    // ================================================================

    // ================================================================
    // 显式赋值接口
    // 在cache duration内对一个key赋值
    // 如果尚未ready，则赋值成功，并激活相关callback
    // 如果有尚未完成的LoadFunction，则当其返回时调用done会无效
    // callback运行在调用set的线程中
    // 是否使用/使用何种异步机制由用户在callback中实现
    // 参数
    // key:               需要赋值的key
    // value:             传入value，用于赋值给key
    //                    value在调用成功后即不可再使用
    // 返回值
    // OK:      表示成功完成赋值
    // BUSY:    仅在GC延迟时出现，未进入处理流程，可以退避重试
    // ERROR:   已经通过set或者LoadFunction赋值过一次，无需重试
    int32_t set(const K& key, V&& value) noexcept {
        ::std::shared_ptr<const V> value_ptr(::std::move(value));
        int32_t ret = set_with_timestamp(key, ::std::move(value_ptr));
        if (unlikely(ret != 0)) {
            value = ::std::move(*value_ptr);
        }
        return ret;
    }

    int32_t set(const K& key, ::std::shared_ptr<const V>&& value) noexcept {
        return set_with_timestamp(key, ::std::move(value));
    }
    // 显式赋值接口
    // ================================================================

    // ================================================================
    // 非阻塞取值接口
    // 获取在cache duration内赋的值
    // 如果尚未ready，则立刻返回空
    // 已经ready，则返回value值
    // 参数
    // key:               需要取值的key
    // error_code:        ready时，接收LoadFunction的返回码
    // value:             ready时，接收LoadFunction的值
    // 返回值
    // OK:      表示成功获取到返回值
    // BUSY:    仅在GC延迟时出现，未进入处理流程，可以退避重试
    // EMPTY:   当前对应key还没有返回值
    int32_t try_get(const K& key, int32_t& error_code, ::std::shared_ptr<const V>& value) noexcept {
        return try_get_with_timestamp(key, error_code, value);
    }
    // 非阻塞取值接口
    // ================================================================

    // 打印状态，调试用
    void log_status() const noexcept {
        LOG(NOTICE) << "cache " << _name << "(" << this << ")"
            << " duration " << _cache_duration_us << " us"
            << " expire " << _epoch_expire_us << " us"
            << " mask " << (void*)_epoch_mask
            << " mask bits " << _epoch_mask_bits;
        for (size_t i = 0; i < SLICE_NUM; ++i) {
            _slices[i].log_status();
        }
    }

private:
    // error code和value的pair，打包作为load结果原子更新
    typedef ::std::pair<int32_t, ::std::shared_ptr<const V>> LoadResult;

    // value callback的单链表节点
    class ValueCallback {
    public:
        ValueCallback(ValueCallbackFunction&& function, ValueCallback* next = nullptr) noexcept:
            _function(::std::move(function)), _task(), _next(next) {}

        ValueCallback(ValueCallbackTask&& task, ValueCallback* next = nullptr) noexcept:
            _function(), _task(::std::move(task)), _next(next) {}

        void next(ValueCallback* new_next) noexcept {
            _next = new_next;
        }

        ValueCallback* next() noexcept {
            return _next;
        }

        void operator()(int32_t error_code, const ::std::shared_ptr<const V>& value) noexcept {
            if (_function) {
                _function(error_code, value);
            } else {
                _task(error_code, value);
            }
        }

        template <typename C, typename std::enable_if<std::is_same<ValueCallbackFunction, C>::value, int>::type = 0>
        C&& move() noexcept {
            return ::std::move(_function);
        }

        template <typename C, typename std::enable_if<std::is_same<ValueCallbackTask, C>::value, int>::type = 0>
        C&& move() noexcept {
            return ::std::move(_task);
        }

    private:
        ValueCallbackFunction _function;
        ValueCallbackTask _task;
        ValueCallback* _next;
    };

    // value callback单链表
    // 创建后可以多次挂载新节点，但是只能被执行一次
    // 执行时会依次调用挂载的value callback
    // 执行后表头置不可用，后续挂载尝试都会失败
    class ValueCallbackChain {
    public:
        ValueCallbackChain() noexcept : _head(nullptr) {}

        ValueCallbackChain(ValueCallbackChain&& other) noexcept :
            _head(other._head.exchange(0, ::std::memory_order_relaxed)) {}

        template <typename C>
        int32_t register_callback(C&& callback) noexcept;

        int32_t run(int32_t error_code, const ::std::shared_ptr<const V>& value) noexcept;

    private:
        static ValueCallback* const SEALED_HEAD;
        ::std::atomic<ValueCallback*> _head;
    };

    class Node {
    public:
        Node() noexcept : _result(nullptr), _callback_chain() {
        }

        Node(const Node& other) noexcept {
            // 不会用到这个构造，原理上也不支持
            CFATAL_LOG(" node copy happen, maybe a bug?");
            abort();
        }

        Node(Node&& other) noexcept :
            _result(other._result.exchange(nullptr, ::std::memory_order_relaxed)),
            _callback_chain(::std::move(other._callback_chain)) {
        }

        ~Node() noexcept {
            const auto result = _result.load(::std::memory_order_relaxed);
            if (result != nullptr) {
                delete result;
            }
            _callback_chain.run(ERROR, DEFAULT_VALUE);
        }

        template <typename C>
        void monitor(C&& callback) noexcept {
            if (unlikely(0 != register_callback(::std::move(callback)))) {
                // 注册失败，表示已经ready，直接执行value callback
                const auto load_result = _result.load(::std::memory_order_acquire);
                if (unlikely(load_result == nullptr)) {
                    // callback无法注册但是value又不存在，不应该有这个状态
                    CFATAL_LOG(" node finished without result, maybe a bug?");
                    abort();
                }
                callback(load_result->first, load_result->second);
            }
        }

        template <typename C>
        void check_and_monitor(C&& callback) noexcept {
            const auto load_result = _result.load(::std::memory_order_acquire);
            if (load_result != nullptr) {
                // 检查时已经ready，直接调用value callback
                callback(load_result->first, load_result->second);
            } else {
                // 还未ready，注册value callback监控node
                monitor(::std::move(callback));
            }
        }

        // ===========================================================
        // 将node设置为ready状态，只能进行一次，后续设置都会失败

        // 设置成功结束，并设置value
        // 调用成功后value不再可用，失败时会留在原地
        int32_t set(::std::shared_ptr<const V>&& value) {
            LoadResult* result = new LoadResult(0, ::std::move(value));
            if (unlikely(0 != set_result(result))) {
                value = ::std::move(result->second);
                delete result;
                return ERROR;
            }
            return OK;
        }

        // 设置失败结束，并设置error_code
        int32_t fail(int32_t error_code) noexcept {
            LoadResult* result = new LoadResult(::std::move(error_code), nullptr);
            if (unlikely(0 != set_result(result))) {
                delete result;
                return ERROR;
            }
            return OK;
        }
        // =============================================================

        int32_t get(int32_t& error_code, ::std::shared_ptr<const V>& value) noexcept {
            const auto result = _result.load(::std::memory_order_relaxed);
            if (result != nullptr) {
                error_code = result->first;
                value = result->second;
                return OK;
            } else {
                return EMPTY;
            }
        }

        template <typename C>
        int32_t register_callback(C&& callback) noexcept {
            return _callback_chain.register_callback(::std::move(callback));
        }

    private:
        static const ::std::shared_ptr<const V> DEFAULT_VALUE;
        ::std::atomic<const LoadResult*> _result;
        ValueCallbackChain _callback_chain;

        int32_t set_result(LoadResult* result) noexcept {
            const LoadResult* expected_result = nullptr;
            if (unlikely(!_result.compare_exchange_strong(
                        expected_result, result, ::std::memory_order_acq_rel))) {
                return ERROR;
            }

            // ready后开始激活callback chain
            if (unlikely(0 != _callback_chain.run(result->first, result->second))) {
                CFATAL_LOG(" value ready when callback chain already run, maybe a bug?");
                abort();
            }
            return OK;
        }
    };

    class Slice {
    public:
        Slice() : _index(0), _visitor_num(0), _epoch(INT64_MIN) {}

        void index(size_t index) noexcept {
            _index = index;
        }

        void enter() noexcept {
            static ::std::memory_order memory_order = (O & TransientCacheOption::SAFE_EXPIRATION) == 0 ?
                ::std::memory_order_release : ::std::memory_order_seq_cst;
            _visitor_num.fetch_add(1, memory_order);
        }

        void leave() noexcept {
            static ::std::memory_order memory_order = (O & TransientCacheOption::SAFE_EXPIRATION) == 0 ?
                ::std::memory_order_release : ::std::memory_order_seq_cst;
            _visitor_num.fetch_sub(1, memory_order);
        }

        int64_t epoch(
            ::std::memory_order memory_order = ::std::memory_order_acquire) const noexcept {
            return _epoch.load(memory_order);;
        }

        int32_t reset_to_epoch(int64_t new_epoch) noexcept {
            static ::std::memory_order store_memory_order = (O & TransientCacheOption::SAFE_EXPIRATION) == 0 ?
                ::std::memory_order_release : ::std::memory_order_seq_cst;
            static ::std::memory_order load_memory_order = (O & TransientCacheOption::SAFE_EXPIRATION) == 0 ?
                ::std::memory_order_acquire : ::std::memory_order_seq_cst;

            // 先设置成不可用
            _epoch.store(INT64_MIN, store_memory_order);

            // 再查看是否还有残存的访问者
            int64_t visitor_num = _visitor_num.load(load_memory_order);
            // 强制过期
            if (unlikely(visitor_num > 0)) {
                if (O & TransientCacheOption::FORCE_EXPIRATION) {
                    LOG(WARNING) << "force clear slice, maybe a bug?";
                } else {
                    return BUSY;
                }
            }

            // 开始清空数据
            _map.clear();

            // 推进到新epoch
            _epoch.store(new_epoch, ::std::memory_order_release);
            return OK;
        }

        template <typename C>
        void get(const K& key, int64_t epoch,
            const LoadFunction& load_function,
            C&& callback) noexcept {
            // 尝试插入目标节点
            auto result = _map.emplace(key, Node());
            auto& node = result.first->second;
            if (result.second) {
                // 第一个到达的pioneer
                node.monitor(::std::move(callback));
                int32_t ret = load_function(key, LoadCallback(*this, node, epoch));
                if (unlikely(0 != ret)) {
                    // 发起load失败，以失败状态结束node
                    node.fail(ret);
                }
            } else {
                // 后续访客，检测节点状态，在ready之后执行callback
                node.check_and_monitor(::std::move(callback));
            }
        }

        template <typename C>
        ValueCallbackFunction get_for_load(const K& key, int64_t epoch,
            C&& callback) noexcept {
            // 尝试插入目标节点
            auto result = _map.emplace(key, Node());
            auto& node = result.first->second;
            if (result.second) {
                // 第一个到达的pioneer
                node.monitor(::std::move(callback));
                LoadCallback load_callback(*this, node, epoch);
                return [load_callback](int32_t error_code,
                        const ::std::shared_ptr<const V>& value) mutable {
                    if (likely(error_code == 0)) {
                        load_callback.done(::std::shared_ptr<const V>(value));
                    } else {
                        load_callback.done(error_code);
                    }
                };
            } else {
                // 后续访客，检测节点状态，在ready之后执行callback
                node.check_and_monitor(::std::move(callback));
                return ValueCallbackFunction();
            }
        }

        int32_t set(const K& key, ::std::shared_ptr<const V>&& value) noexcept {
            // 尝试插入目标节点
            auto result = _map.emplace(key, Node());
            auto& node = result.first->second;
            if (result.second) {
                // 第一个到达的pioneer
                return node.set(::std::move(value));
            } else {
                return ERROR;
            }
        }

        int32_t try_get(const K& key, int32_t& error_code,
            ::std::shared_ptr<const V>& value) noexcept {
            // 尝试插入目标节点
            auto it = _map.find(key);
            if (it != _map.end()) {
                return it->second.get(error_code, value);
            } else {
                return EMPTY;
            }
        }

        void log_status() const noexcept {
            LOG(NOTICE) << "slice " << _index
                << " epoch " << _epoch.load(::std::memory_order_relaxed)
                << " size " << _map.size()
                << " visitor " << _visitor_num.load(::std::memory_order_relaxed);
        }

    private:
        size_t _index;
        ::std::atomic<int64_t> _visitor_num;
        ::std::atomic<int64_t> _epoch;
        ::tbb::concurrent_unordered_map<K, Node, H> _map;
    };

    class SliceVisitor {
    public:
        SliceVisitor() noexcept : _slice(nullptr) {}

        SliceVisitor(Slice& slice) noexcept : _slice(&slice) {_slice->enter();}

        SliceVisitor(SliceVisitor&& other) noexcept : _slice(other._slice) {
            other._slice = nullptr;
        }

        ~SliceVisitor() noexcept {
            if (_slice != nullptr) {
                _slice->leave();
            }
        }

        inline Slice* slice() noexcept {
            return _slice;
        }

    private:
        Slice* _slice;
    };

    void stop_gc() noexcept {
        _running.store(false, ::std::memory_order_relaxed);
        if (_gc_thread.joinable()) {
            _gc_thread.join();
        }
    }

    void gc_thread() noexcept;

    int64_t gc(int64_t now = ::base::gettimeofday_us()) noexcept;

    template <typename C>
    int32_t get_with_timestamp(const K& key, C&& calback,
        int64_t now = ::base::gettimeofday_us()) noexcept;

    int32_t set_with_timestamp(const K& key, ::std::shared_ptr<const V>&& value,
        int64_t now = ::base::gettimeofday_us()) noexcept;

    int32_t try_get_with_timestamp(const K& key, int32_t& error_code,
        ::std::shared_ptr<const V>& value, int64_t now = ::base::gettimeofday_us()) noexcept;

    int64_t epoch_for_timestamp(int64_t timestamp_us) noexcept {
        return timestamp_us & _epoch_mask;
    }

    int64_t slice_index_for_epoch(int64_t epoch) noexcept {
        return (epoch >> _epoch_mask_bits) & SLICE_MASK;
    }

    SliceVisitor slice_for_epoch(int64_t epoch) noexcept {
        static ::std::memory_order memory_order = (O & TransientCacheOption::SAFE_EXPIRATION) == 0 ?
            ::std::memory_order_acquire : ::std::memory_order_seq_cst;
        int64_t now_slice_index = (epoch >> _epoch_mask_bits) & SLICE_MASK;
        Slice& slice = _slices[now_slice_index];
        SliceVisitor visitor(slice);
        int64_t slice_epoch = slice.epoch(memory_order);
        return slice_epoch == epoch ? ::std::move(visitor) : SliceVisitor();
    }

    static const int32_t SLICE_NUM = 4;
    static const int32_t SLICE_MASK = SLICE_NUM - 1;
    
    ::std::string _name;
    LoadFunction _load_function;
    Slice _slices[SLICE_NUM];
    int64_t _cache_duration_us;
    int64_t _epoch_expire_us;
    uint64_t _epoch_mask;
    uint64_t _epoch_mask_bits;

    ::std::atomic<bool> _running;
    ::std::thread _gc_thread;
};

template <typename K, typename V, typename H, uint32_t O>
const ::std::shared_ptr<const V> TransientCache<K, V, H, O>::Node::DEFAULT_VALUE;

// =============================================================================
// TransientCache::LoadCallback begin
template <typename K, typename V, typename H, uint32_t O>
void TransientCache<K, V, H, O>::LoadCallback::done(::std::shared_ptr<const V>&& value) noexcept {
    static ::std::memory_order memory_order = (O & TransientCacheOption::SAFE_EXPIRATION) == 0 ?
        ::std::memory_order_acquire : ::std::memory_order_seq_cst;
    // 先标记访问状态
    SliceVisitor visitor(_slice);
    // 之后读取当前的epoch
    int64_t slice_epoch = _slice.epoch(memory_order);
    if (unlikely(_epoch != slice_epoch)) {
        // slice已经推进到或者推进中
        CWARNING_LOG(" ignore legacy callback at epoch %" PRId64 " < %" PRId64 "",
            _epoch, slice_epoch);
    } else {
        // 赋值并标记节点结束
        _node.set(::std::move(value));
    }

    if (unlikely(O & TransientCacheOption::SAFE_DESTRUCTION)) {
        _slice.leave();
    }
}

template <typename K, typename V, typename H, uint32_t O>
void TransientCache<K, V, H, O>::LoadCallback::done(int32_t error_code) noexcept {
    static ::std::memory_order memory_order = (O & TransientCacheOption::SAFE_EXPIRATION) == 0 ?
        ::std::memory_order_acquire : ::std::memory_order_seq_cst;
    // 先标记访问状态
    SliceVisitor visitor(_slice);
    // 之后读取当前的epoch
    int64_t slice_epoch = _slice.epoch(memory_order);
    if (unlikely(_epoch != slice_epoch)) {
        // slice已经推进到或者推进中
        CWARNING_LOG(" ignore legacy callback at epoch %" PRId64 " < %" PRId64 "",
            _epoch, slice_epoch);
    } else {
        // 赋值并标记节点结束
        _node.fail(error_code);
    }

    if (unlikely(O & TransientCacheOption::SAFE_DESTRUCTION)) {
        _slice.leave();
    }
}
// TransientCache::LoadCallback end
// =============================================================================

// =============================================================================
// TransientCache begin
template <typename K, typename V, typename H, uint32_t O>
TransientCache<K, V, H, O>::TransientCache(const LoadFunction& load_function, int64_t cache_duration_us,
    uint32_t amplification, double jitter, const std::string& name) noexcept : _name(name),
    _load_function(load_function), _cache_duration_us(cache_duration_us), _running(true) {
    for (size_t i = 0; i < SLICE_NUM; ++i) {
        _slices[i].index(i);
    }

    // cache duration放大倍数后作为slice的epoch
    int64_t epoch_duration_us = _cache_duration_us * amplification;

    // 根据epoch大小，生成mask和bits
    _epoch_mask = 0xFFFFFFFFFFFFFFFFL;
    _epoch_mask_bits = 0;
    while (0 != (epoch_duration_us & _epoch_mask)) {
        ++_epoch_mask_bits;
        _epoch_mask <<= 1;
    }

    // epoch过期时间按照2倍epoch + cache duration计算，增加jitter避免所有cache同时淘汰
    // jitter[0.0, 1.0]对应增加[0, epoch - cache duration]
    int64_t jitter_us = jitter * ((uint64_t)::std::random_device()() % ((1L << _epoch_mask_bits) - _cache_duration_us));
    _epoch_expire_us = (2L << _epoch_mask_bits) + _cache_duration_us + jitter_us;

    // 通过gc初始化epoch到当前时间 - _cache_duration_us
    // 往前偏移是为了保证最初的get如果在slice边界，获取上一个slice是成功的
    gc(::base::gettimeofday_us() - _cache_duration_us);
    // 启动gc thread后续定期gc
    _gc_thread = ::std::thread(&TransientCache::gc_thread, this);
}

template <typename K, typename V, typename H, uint32_t O>
void TransientCache<K, V, H, O>::gc_thread() noexcept {
    auto next_gc_us = gc();
    while (_running.load(::std::memory_order_relaxed)) {
        if (next_gc_us > 0) {
            // 每次sleep不超过100ms，避免长期无法析构
            auto now = ::base::gettimeofday_us();
            int64_t time_to_next_gc_us = next_gc_us - now;
            if (time_to_next_gc_us > 100000) {
                usleep(100000);
                continue;
            } else if (time_to_next_gc_us > 0) {
                usleep(time_to_next_gc_us);
                continue;
            }
        } else if (next_gc_us == INT64_MIN) {
            // 上次gc受阻，交出cpu
            sched_yield();
        }
        log_status();
        next_gc_us = gc();
    }
}

template <typename K, typename V, typename H, uint32_t O>
int64_t TransientCache<K, V, H, O>::gc(int64_t now) noexcept {
    int64_t now_epoch = epoch_for_timestamp(now);
    int64_t now_slice_index = slice_index_for_epoch(now_epoch);
    int64_t next_gc_us = INT64_MAX;
    for (size_t i = 0; i < SLICE_NUM; ++i) {
        auto& slice = _slices[i];
        int64_t slice_epoch = slice.epoch();
        int64_t expire_timestamp_us = slice_epoch + _epoch_expire_us;
        if (expire_timestamp_us < now) {
            int64_t slice_beyond = now_slice_index > i ?
                (i + SLICE_NUM - now_slice_index) : (i - now_slice_index);
            int64_t old_slice_epoch = slice_epoch;
            slice_epoch = now_epoch + (slice_beyond << _epoch_mask_bits);
            int64_t begin = ::base::gettimeofday_us();
            if (likely(OK == slice.reset_to_epoch(slice_epoch))) {
                // 推进成功，更新过期时间
                expire_timestamp_us = slice_epoch + _epoch_expire_us;
                int64_t end = ::base::gettimeofday_us();
                LOG(NOTICE) << "cache " << _name << "(" << (void*)this << ")"
                    << " move slice " << i
                    << " from " << old_slice_epoch << " to " << slice_epoch
                    << " use " << end - begin << " us";
            } else {
                // 因为还有访问者，无法清空并推进，设置特殊标记，yield之后调度
                expire_timestamp_us = INT64_MIN;
            }
        }
        // 迭代取最近的过期时间
        if (expire_timestamp_us < next_gc_us) {
            next_gc_us = expire_timestamp_us;
        }
    }
    return next_gc_us;
}

template <typename K, typename V, typename H, uint32_t O>
int32_t TransientCache<K, V, H, O>::try_get_with_timestamp(
    const K& key, int32_t& error_code, ::std::shared_ptr<const V>& value,
    int64_t timestamp_us) noexcept {
    // 取得当前epoch
    auto epoch = epoch_for_timestamp(timestamp_us - _cache_duration_us);
    auto next_epoch = epoch_for_timestamp(timestamp_us);
    auto visitor = slice_for_epoch(epoch);
    auto slice = visitor.slice();
    if (unlikely(slice == nullptr)) {
        // slice还没有推进到目标epoch
        CWARNING_LOG(" slice not ready duration is too short for gc ?");
        return BUSY;
    }

    int32_t ret = slice->try_get(key, error_code, value);
    if (likely(epoch == next_epoch) || ret == OK) {
        return ret;
    }

    auto next_visitor = slice_for_epoch(next_epoch);
    auto next_slice = next_visitor.slice();
    if (unlikely(next_slice == nullptr)) {
        // slice还没有推进到目标epoch
        CWARNING_LOG(" slice not ready duration is too short for gc ?");
        return BUSY;
    }
    
    return next_slice->try_get(key, error_code, value);
}

template <typename K, typename V, typename H, uint32_t O>
int32_t TransientCache<K, V, H, O>::set_with_timestamp(
    const K& key, ::std::shared_ptr<const V>&& value, int64_t timestamp_us) noexcept {
    // 取得当前epoch
    auto epoch = epoch_for_timestamp(timestamp_us);
    auto visitor = slice_for_epoch(epoch);
    auto slice = visitor.slice();
    if (unlikely(slice == nullptr)) {
        // slice还没有推进到目标epoch
        CWARNING_LOG(" slice not ready duration is too short for gc ?");
        return BUSY;
    }

    return slice->set(key, ::std::move(value));
}

template <typename K, typename V, typename H, uint32_t O>
template <typename C>
int32_t TransientCache<K, V, H, O>::get_with_timestamp(
    const K& key, C&& callback, int64_t timestamp_us) noexcept {
    // 前推cache duration取得base epoch
    auto epoch = epoch_for_timestamp(timestamp_us - _cache_duration_us);
    auto visitor = slice_for_epoch(epoch);
    auto slice = visitor.slice();
    if (unlikely(slice == nullptr)) {
        // slice还没有推进到目标epoch
        CWARNING_LOG(" slice not ready duration is too short for gc ?");
        return BUSY;
    }

    // 取得当前epoch
    auto next_epoch = epoch_for_timestamp(timestamp_us);
    if (likely(next_epoch == epoch)) {
        // 和base是同一个epoch，直接get
        slice->get(key, epoch, _load_function, ::std::move(callback));
        return OK;
    }

    // 否则取得下一个slice
    auto next_visitor = slice_for_epoch(next_epoch);
    auto next_slice = next_visitor.slice();
    if (unlikely(next_slice == nullptr)) {
        // slice还没有推进到目标epoch
        CWARNING_LOG(" slice not ready duration is too short for gc ?");
        return BUSY;
    }

    // 首先操作base slice，但如果自己是pioneer时并不执行load
    auto next_function = slice->get_for_load(key, epoch, ::std::move(callback));
    if (next_function) {
        // pioneer继续操作next slice，完成时传入base slice的级联回调
        next_slice->get(key, next_epoch, _load_function, ::std::move(next_function));
    }
    return OK;
}
// TransientCache end
// =============================================================================

// =============================================================================
// TransientCache::ValueCallbackChain begin
template <typename K, typename V, typename H, uint32_t O>
typename TransientCache<K, V, H, O>::ValueCallback* const
TransientCache<K, V, H, O>::ValueCallbackChain::SEALED_HEAD =
    reinterpret_cast<ValueCallback*>(0xFFFFFFFFFFFFFFFFL);

template <typename K, typename V, typename H, uint32_t O>
template <typename C>
int32_t TransientCache<K, V, H, O>::ValueCallbackChain::register_callback(
    C&& callback) noexcept {
    ValueCallback* head = _head.load(::std::memory_order_relaxed);
    ValueCallback* new_callback = new ValueCallback(::std::move(callback), head);

    while (true) {
        if (unlikely(head == SEALED_HEAD)) {
            // callback已经run过了，把function还给外层
            callback = new_callback->template move<C>();
            // 释放没挂上去的节点
            delete new_callback;
            return ERROR;
        }

        if (likely(_head.compare_exchange_weak(head, new_callback,
            ::std::memory_order_acq_rel))) {
            // 挂载成功，退出循环
            return OK;
        }

        // 挂载失败时更新head，再重试一次
        new_callback->next(head);
    }
}

template <typename K, typename V, typename H, uint32_t O>
int32_t TransientCache<K, V, H, O>::ValueCallbackChain::run(
    int32_t error_code, const ::std::shared_ptr<const V>& value) noexcept {
    ValueCallback* head = _head.load(::std::memory_order_relaxed);
    do {
        if (unlikely(head == SEALED_HEAD)) {
            // 已经运行过了，返回失败
            return ERROR;
        }
    // 尝试锁定表头
    } while (unlikely(!_head.compare_exchange_weak(head,
                SEALED_HEAD, ::std::memory_order_acq_rel)));

    // 取出实际的指针
    while (head != nullptr) {
        (*head)(error_code, value);
        ValueCallback* next = head->next();
        delete head;
        head = next;
    }
    return OK;
}
// TransientCache::ValueCallbackChain end
// =============================================================================

}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif  // BAIDU_FEED_MLARCH_BABYLON_CACHE_H
