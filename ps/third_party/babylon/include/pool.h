#ifndef BAIDU_FEED_MLARCH_BABYLON_POOL_H
#define BAIDU_FEED_MLARCH_BABYLON_POOL_H

#include <stack>
#include <mutex>

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

// 对象池，用于解决临时大对象反复创建问题
// 可以的情况下优先使用thread_local，性能最好
// 受限于bthread和某些lib限制的情况下，可以使用对象池
// 使用见单测test/test_pool.cpp
// 完整压测bench/bench_pool.cpp
template <typename T>
class ObjectPool {
private:
    struct Node;
public:
    // 池化的对象，托管了对象释放回池的操作，可以理解成一个unique_ptr
    class PooledObject {
    public:
        // 只能默认构造
        inline PooledObject() = default;
        // 或者移动构造
        inline PooledObject(PooledObject&& other) noexcept;
        // 也可以移动赋值
        inline PooledObject& operator=(PooledObject&& other) noexcept;
        // 销毁对象，实际会将对象释放回池
        inline ~PooledObject() noexcept;
        // 仿指针
        inline T& operator*() noexcept;
        inline const T& operator*() const noexcept;
        inline T* operator->() noexcept;
        inline const T* operator->() const noexcept;
        // 取指针
        inline T* get() noexcept;
        inline const T* get() const noexcept;
        // 交换两个池化对象内容
        inline void swap(PooledObject& other) noexcept;
        // 主动释放对象
        inline void release() noexcept;
        // 获取对象所属ObjectPool，容量不足时临时创建的伪PooledObject
        // 实际不属于任何ObjectPool，会返回nullptr
        // 已经不再持有对象的PooledObject，也会返回nullptr
        inline ObjectPool* pool() noexcept;

    private:
        // 构建非池化对象，析构时会直接析构对象
        inline PooledObject(T* object) noexcept;
        // 构建池化对象，记录池信息和池节点，用于析构时释放回池
        inline PooledObject(ObjectPool* pool, Node* node) noexcept;
        // 取消对内部池化对象的托管，并返回池节点，用于批量释放
        // 未持有池化节点的PooledObject，返回nullptr
        inline Node* de_manage() noexcept;

        T* _object {nullptr};
        ObjectPool* _pool {nullptr};
        Node* _node {nullptr};

        friend class ObjectPool;
    };
    
    // 对象创建function，用于自定义对象构建，以及支持不可默认构造的对象
    typedef ::std::function<T*() noexcept> CreateFunction;

    // ==================================================================
    // 构造函数
    // 创建一个对象池，并默认构建一些池化对象待命
    // 参数
    // create_function:   构建对象的工厂函数
    //                    默认使用T()构建
    // reserve:           预先构建好的对象个数
    //                    默认使用-babylon_pool_reserve
    // cache_per_thread:  每个线程可以局部缓存的对象个数
    //                    默认使用-babylon_pool_cache_per_thread
    // reserve_global:    保留只用于全局池的对象个数，不会用于线程缓存
    //                    默认使用-babylon_pool_reserve_global
    inline ObjectPool() noexcept;
    inline ObjectPool(size_t num) noexcept;
    inline ObjectPool(size_t num, size_t cache_per_thread, size_t reserve_global) noexcept;
    inline ObjectPool(const CreateFunction& create_function) noexcept;
    inline ObjectPool(const CreateFunction& create_function, size_t num) noexcept;
    inline ObjectPool(const CreateFunction& create_function,
        size_t num, size_t cache_per_thread, size_t reserve_global) noexcept;
    // 构造函数
    // ==================================================================

    // ==================================================================
    // 单例配置接口
    // 参数同构造函数，用于定制单例
    // 只有第一次调用生效，只有在调用instance前有效，否则单例为默认构造
    inline static int32_t config() noexcept;
    inline static int32_t config(size_t num) noexcept;
    inline static int32_t config(size_t num, size_t cache_per_thread, size_t reserve_global) noexcept;
    inline static int32_t config(const CreateFunction& create_function) noexcept;
    inline static int32_t config(const CreateFunction& create_function, size_t num) noexcept;
    inline static int32_t config(const CreateFunction& create_function,
        size_t num, size_t cache_per_thread, size_t reserve_global) noexcept;
    // 单例配置接口
    // ==================================================================

    // 方便的单例，适用于大多简单场景，可以避免程序中传递pool
    inline static ObjectPool& instance() noexcept;

    // 获取对象接口
    // 返回包装过的对象，当池内还有剩余时，使用池化对象
    // 否则现场创建一个
    inline PooledObject get() noexcept;

    // 批量释放对象
    // 传入池化对象迭代器
    // 内部通过打包后一轮原子操作释放回池来减少竞争
    template <typename IT>
    inline void release(IT begin, IT end) noexcept;

    // 打印统计信息
    // 后续应使用暴露bvar的方式进行统计数据
    inline __attribute__((deprecated)) void log_statistic() const noexcept;
    // 将统计量暴露到bvar，使用babylon_pool_$name前缀
    // 在开启FLAGS_babylon_pool_statistic的情况下
    // 默认会expose(to_string(this));
    // 用于观测设置的各项参数是否合理
    // 追求local尽量多，否则可以尝试调大线程缓存
    // 追求new足够少，否则可以尝试调大池大小
    inline void expose(const ::std::string& name) noexcept;

private:
    struct Node {
        inline uint64_t advance() noexcept;

        T* object {nullptr};
        uint64_t next {0xFFFFFFFFFFFFFFFFL};
        uint64_t versioned_index {0};
        bool cacheable {false};
    };

    struct NodeChain {
        inline NodeChain() = default;
        inline NodeChain(Node* node) noexcept;
        inline void add(Node* node) noexcept;

        Node* head {nullptr};
        Node* tail {nullptr};
    };

    // 从versioned_index中获取index
    inline static size_t get_index(uint64_t value) noexcept;
    inline Node* try_pop() noexcept;
    inline Node* try_pop_local() noexcept;
    inline Node* try_pop_from(::std::atomic<uint64_t>& head) noexcept;
    inline void push(Node* node) noexcept;
    inline int32_t push_local(Node* node) noexcept;
    inline static void push_into(::std::atomic<uint64_t>& head, Node* node) noexcept;
    inline static void push_into(::std::atomic<uint64_t>& head, NodeChain& chain) noexcept;

    static ::std::atomic<bool> _s_init_flag;
    static ::std::once_flag _s_init_once_flag;
    static ::std::unique_ptr<ObjectPool> _s_pool;
    static ::std::atomic<int32_t> _s_max_cache_index;

    CreateFunction _create_function;
    ::std::atomic<uint64_t> _cacheable_head {0xFFFFFFFFFFFFFFFFL};
    ::std::atomic<uint64_t> _head {0xFFFFFFFFFFFFFFFFL};
    typename ::std::conditional<
        ::std::is_default_constructible<T>::value,
        ::std::vector<T>,
        int32_t>::type _objects;
    ::std::vector<::std::shared_ptr<T>> _object_ptrs;
    ::std::vector<Node> _nodes;
    int32_t _cache_index {_s_max_cache_index.fetch_add(1, ::std::memory_order_relaxed)};
    size_t _cache_per_thread;
    size_t _reserve_global;
    friend class PooledObject;
};

}  // babylon
}  // mlarch
}  // feed
}  // baidu
#endif //BAIDU_FEED_MLARCH_BABYLON_POOL_H

#include "pool.hpp"
