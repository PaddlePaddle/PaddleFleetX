#ifndef BAIDU_FEED_MLARCH_BABYLON_POOL_HPP
#define BAIDU_FEED_MLARCH_BABYLON_POOL_HPP

#include <atomic>
#include <gflags/gflags.h>
#include "expect.h"
#include "pool.h"

DECLARE_bool(babylon_pool_statistic);
DECLARE_uint64(babylon_pool_reserve);
DECLARE_uint64(babylon_pool_cache_per_thread);
DECLARE_uint64(babylon_pool_reserve_global);
DECLARE_uint64(babylon_pool_create_thread_num);

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

///////////////////////////////////////////////////////////////////////////////
// ObjectPool::PooledObject begin
template <typename T>
ObjectPool<T>::PooledObject::PooledObject(T* object) noexcept :
    _object(object), _pool(nullptr), _node(nullptr) {}

template <typename T>
ObjectPool<T>::PooledObject::PooledObject(ObjectPool* pool, Node* node) noexcept :
    _object(node->object), _pool(pool), _node(node) {}

template <typename T>
ObjectPool<T>::PooledObject::PooledObject(PooledObject&& other) noexcept :
    _object(other._object), _pool(other._pool), _node(other._node) {
    other._object = nullptr;
    other._pool = nullptr;
    other._node = nullptr;
}

template <typename T>
ObjectPool<T>::PooledObject::~PooledObject() noexcept {
    release();
}

template <typename T>
typename ObjectPool<T>::PooledObject& ObjectPool<T>::PooledObject::operator=(PooledObject&& other) noexcept {
    swap(other);
    other.release();
    return *this;
}

template <typename T>
T& ObjectPool<T>::PooledObject::operator*() noexcept {
    return *_object;
}

template <typename T>
const T& ObjectPool<T>::PooledObject::operator*() const noexcept {
    return *_object;
}

template <typename T>
T* ObjectPool<T>::PooledObject::operator->() noexcept {
    return _object;
}

template <typename T>
const T* ObjectPool<T>::PooledObject::operator->() const noexcept {
    return _object;
}

template <typename T>
T* ObjectPool<T>::PooledObject::get() noexcept {
    return _object;
}

template <typename T>
const T* ObjectPool<T>::PooledObject::get() const noexcept {
    return _object;
}

template <typename T>
void ObjectPool<T>::PooledObject::swap(PooledObject& other) noexcept {
    ::std::swap(_object, other._object);
    ::std::swap(_pool, other._pool);
    ::std::swap(_node, other._node);
}

template <typename T>
void ObjectPool<T>::PooledObject::release() noexcept {
    if (likely(_pool != nullptr)) {
        _pool->push(_node);
    } else if (unlikely(_object != nullptr)) {
        delete _object;
    }
    _object = nullptr;
    _pool = nullptr;
    _node = nullptr;
}

template <typename T>
ObjectPool<T>* ObjectPool<T>::PooledObject::pool() noexcept {
    return _pool;
}

template <typename T>
typename ObjectPool<T>::Node* ObjectPool<T>::PooledObject::de_manage() noexcept {
    if (likely(_pool != nullptr)) {
        auto node = _node;
        _object = nullptr;
        _pool = nullptr;
        _node = nullptr;
        return node;
    }
    return nullptr;
}
// ObjectPool::PooledObject end
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// ObjectPool::Node begin
template <typename T>
uint64_t ObjectPool<T>::Node::advance() noexcept {
    versioned_index += 0x100000000;
    return versioned_index;
}
// ObjectPool::Node end
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// ObjectPool::NodeChain begin
template <typename T>
ObjectPool<T>::NodeChain::NodeChain(Node* node) noexcept :
    head(node), tail(node) {}

template <typename T>
void ObjectPool<T>::NodeChain::add(Node* node) noexcept {
    if (head == nullptr) {
        head = tail = node;
    } else {
        node->next = head->advance();
        head = node;
    }
}
// ObjectPool::NodeChain end
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// ObjectPool begin
template <typename T>
ObjectPool<T>::ObjectPool() noexcept :
    ObjectPool(FLAGS_babylon_pool_reserve) {}

template <typename T>
ObjectPool<T>::ObjectPool(size_t num) noexcept :
    ObjectPool(num, FLAGS_babylon_pool_cache_per_thread, FLAGS_babylon_pool_reserve_global) {}

template <typename T>
ObjectPool<T>::ObjectPool(size_t num, size_t cache_per_thread, size_t reserve_global) noexcept :
    _create_function([](){return new (::std::nothrow) T;}),
    _objects(num), _object_ptrs(0),
    _nodes(num), _cache_per_thread(cache_per_thread), _reserve_global(reserve_global) {
    NodeChain cacheable_chain;
    NodeChain non_cacheable_chain;
    for (size_t i = 0; i < num; ++i) {
        auto& object = _objects[i];
        auto& node = _nodes[i];
        node.object = &object;
        node.versioned_index = i;
        if (_cache_per_thread > 0 && i >= _reserve_global) {
            node.cacheable = true;
            cacheable_chain.add(&node);
        } else {
            node.cacheable = false;
            non_cacheable_chain.add(&node);
        }
    }
    if (non_cacheable_chain.head != nullptr) {
        push_into(_head, non_cacheable_chain);
    }
    if (cacheable_chain.head != nullptr) {
        push_into(_cacheable_head, cacheable_chain);
    }
    if (FLAGS_babylon_pool_statistic) {
        expose(::std::to_string(reinterpret_cast<uint64_t>(this)));
    }
}

template <typename T>
ObjectPool<T>::ObjectPool(const CreateFunction& create_function) noexcept :
    ObjectPool(create_function, FLAGS_babylon_pool_reserve) {}

template <typename T>
ObjectPool<T>::ObjectPool(const CreateFunction& create_function, size_t num) noexcept :
    ObjectPool(create_function, num, FLAGS_babylon_pool_cache_per_thread,
        FLAGS_babylon_pool_reserve_global) {}

template <typename T>
ObjectPool<T>::ObjectPool(const CreateFunction& create_function,
    size_t num, size_t cache_per_thread, size_t reserve_global) noexcept :
    _create_function(create_function), _objects(0),
    _object_ptrs(num), _nodes(num),
    _cache_per_thread(cache_per_thread), _reserve_global(reserve_global) {
    for (size_t i = 0; i < num; ++i) {
        auto& object_ptr = _object_ptrs[i];
        object_ptr.reset(_create_function());
        auto& node = _nodes[i];
        node.object = object_ptr.get();
        node.versioned_index = i;
        if (_cache_per_thread > 0 && i >= _reserve_global) {
            node.cacheable = true;
            push_into(_cacheable_head, &node);
        } else {
            node.cacheable = false;
            push_into(_head, &node);
        }
    }
    if (FLAGS_babylon_pool_statistic) {
        expose(::std::to_string(reinterpret_cast<uint64_t>(this)));
    }
}

template <typename T>
int32_t ObjectPool<T>::config() noexcept {
    return config(FLAGS_babylon_pool_reserve);
}

template <typename T>
int32_t ObjectPool<T>::config(size_t num) noexcept {
    return config(num, FLAGS_babylon_pool_cache_per_thread,
        FLAGS_babylon_pool_reserve_global);
}

template <typename T>
int32_t ObjectPool<T>::config(size_t num, size_t cache_per_thread,
    size_t reserve_global) noexcept {
    int32_t ret = 1;
    if (!_s_init_flag.load(::std::memory_order_relaxed)) {
        ::std::call_once(_s_init_once_flag,
            [&ret, num, cache_per_thread, reserve_global]() mutable noexcept {
                _s_pool.reset(new ObjectPool(num, cache_per_thread, reserve_global));
                ret = 0;
        });
    }
    return ret;
}

template <typename T>
int32_t ObjectPool<T>::config(const CreateFunction& create_function) noexcept {
    return config(create_function, FLAGS_babylon_pool_reserve);
}

template <typename T>
int32_t ObjectPool<T>::config(const CreateFunction& create_function, size_t num) noexcept {
    return config(create_function, num, FLAGS_babylon_pool_cache_per_thread,
        FLAGS_babylon_pool_reserve_global);
}

template <typename T>
int32_t ObjectPool<T>::config(const CreateFunction& create_function,
    size_t num, size_t cache_per_thread, size_t reserve_global) noexcept {
    int32_t ret = 1;
    if (!_s_init_flag.load(::std::memory_order_relaxed)) {
        ::std::call_once(_s_init_once_flag,
            [&ret, &create_function, num, cache_per_thread, reserve_global]() mutable noexcept {
                _s_pool.reset(new ObjectPool(create_function,
                        num, cache_per_thread, reserve_global));
                ret = 0;
        });
    }
    return ret;
}

template <typename T>
ObjectPool<T>& ObjectPool<T>::instance() noexcept {
    config();
    return *_s_pool;
}

template <typename T>
typename ObjectPool<T>::PooledObject ObjectPool<T>::get() noexcept {
    auto node = try_pop();
    if (unlikely(node == nullptr)) {
        return PooledObject(_create_function());
    }
    return PooledObject(this, node);
}

template <typename T>
void ObjectPool<T>::log_statistic() const noexcept {
}

template <typename T>
void ObjectPool<T>::expose(const ::std::string& name) noexcept {
}

template <typename T>
size_t ObjectPool<T>::get_index(uint64_t value) noexcept {
    return value & 0xFFFFFFFF;
}

template <typename T>
typename ObjectPool<T>::Node* ObjectPool<T>::try_pop() noexcept {
    Node* node;
    if (_cache_per_thread > 0) {
        node = try_pop_local();
        if (node != nullptr) {
            return node;
        }
        node = try_pop_from(_cacheable_head);
        if (node != nullptr) {
            return node;
        }
    }
    node = try_pop_from(_head);
    return node;
}

template <typename T>
typename ObjectPool<T>::Node* ObjectPool<T>::try_pop_local() noexcept {
    return nullptr;
}

template <typename T>
typename ObjectPool<T>::Node* ObjectPool<T>::try_pop_from(::std::atomic<uint64_t>& head) noexcept {
    auto current_head = head.load(::std::memory_order_acquire);
    Node* node;
    uint64_t next;
    do {
        if (unlikely(current_head == 0xFFFFFFFFFFFFFFFFL)) {
            node = nullptr;
            break;
        }
        node = &_nodes[get_index(current_head)];
        next = node->next;
    } while (!head.compare_exchange_weak(current_head, next, ::std::memory_order_acq_rel));
    return node;
}

template <typename T>
void ObjectPool<T>::push(Node* node) noexcept {
    if (node->cacheable) {
        if (0 == push_local(node)) {
            return;
        }
        push_into(_cacheable_head, node);
    } else {
        push_into(_head, node);
    }
}

template <typename T>
int32_t ObjectPool<T>::push_local(Node* node) noexcept {
    return 1;
}

template <typename T>
void ObjectPool<T>::push_into(::std::atomic<uint64_t>& head, Node* node) noexcept {
    NodeChain chain(node);
    push_into(head, chain);
}

template <typename T>
void ObjectPool<T>::push_into(::std::atomic<uint64_t>& head, NodeChain& chain) noexcept {
    auto versioned_index = chain.head->advance();
    auto current_head = head.load(::std::memory_order_relaxed);
    do {
        chain.tail->next = current_head;
    } while (!head.compare_exchange_weak(current_head, versioned_index,
            ::std::memory_order_acq_rel));
}

//template <typename T, typename IT>
template <typename T>
template <typename IT>
void ObjectPool<T>::release(IT begin, IT end) noexcept {
    NodeChain cacheable_chain;
    NodeChain non_cacheable_chain;
    while (begin != end) {
        if (unlikely(begin->pool() != this)) {
            begin->release();
            continue;
        }

        auto node = begin->de_manage();
        if (unlikely(node == nullptr)) {
            begin->release();
            continue;
        }

        if (!node->cacheable) {
            non_cacheable_chain.add(node);
        } else {
            cacheable_chain.add(node);
        }
        ++begin;
    }

    if (likely(non_cacheable_chain.head != nullptr)) {
        push_into(_head, non_cacheable_chain);
    }
    if (unlikely(cacheable_chain.head != nullptr)) {
        push_into(_cacheable_head, cacheable_chain);
    }
}

template <typename T>
::std::atomic<int32_t> ObjectPool<T>::_s_max_cache_index(0);

template <typename T>
::std::once_flag ObjectPool<T>::_s_init_once_flag;

template <typename T>
::std::atomic<bool> ObjectPool<T>::_s_init_flag(false);

template <typename T>
::std::unique_ptr<ObjectPool<T>> ObjectPool<T>::_s_pool(nullptr);
// ObjectPool end
///////////////////////////////////////////////////////////////////////////////

}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif //BAIDU_FEED_MLARCH_BABYLON_POOL_HPP
