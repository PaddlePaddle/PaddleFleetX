#ifndef BAIDU_FEED_MLARCH_BABYLON_BUFFER_H
#define BAIDU_FEED_MLARCH_BABYLON_BUFFER_H

#include <list>
#include <string.h>
#include <inttypes.h>
#include <gflags/gflags.h>
#include <baidu/feed/mlarch/babylon/expect.h>

DECLARE_uint64(babylon_buffer_block_size);

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

// 静态内存缓冲区，在【静态】【大】词典场景下
// 零散写入构建后，进入只读状态，ReadOnlyBuffer可以提供连续内存管理
// 相关方法都不是线程安全的
class ReadOnlyBuffer {
public:
    ReadOnlyBuffer(size_t block_size = FLAGS_babylon_buffer_block_size) :
        _block_size(block_size) {}

    // ========================================================================
    // 创建'\0'结尾的字符串buffer
    // 从c_str拷贝length + 1的内容
    // 返回拷贝结果对应的起始指针
    // 返回结果在release或者析构后失效
    inline const char* create_cstring_buffer(const ::std::string& c_str) noexcept {
        return create_cstring_buffer(c_str.c_str(), c_str.length());
    }
    inline const char* create_cstring_buffer(const char* c_str, size_t length) noexcept {
        // 空串特殊处理
        if (unlikely(length == 0)) {
            return "";
        }
        // 拷贝并补充'\0'结尾
        char* buffer_start = allocate_buffer(length + 1);
        memcpy(buffer_start, c_str, length);
        buffer_start[length] = '\0';
        return buffer_start;
    }
    // ========================================================================
    
    // ========================================================================
    // 创建二进制buffer
    // 从buffer中拷贝buffer_size的内容
    // 返回拷贝结果对应的起始指针
    // 返回结果在release或者析构后失效
    inline const char* create_buffer(const ::std::string& buffer) noexcept {
        return create_buffer(buffer.data(), buffer.size());
    }
    const char* create_buffer(const char* buffer, size_t buffer_size) noexcept {
        auto buffer_start = allocate_buffer(buffer_size);
        memcpy(buffer_start, buffer, buffer_size);
        return buffer_start;
    }
    // ========================================================================

    //inline char* create_fixed_buffer(size_t size) noexcept {
    //    auto buffer_start = allocate_buffer(size);
    //    return buffer_start;
    //}
    
    // 释放所有buffer中的内存
    // 之前从buffer获取的指针全部失效
    inline void clear() noexcept {
        _blocks.clear();
        _size_in_block = 0;
    }

private:
    char* allocate_buffer(size_t buffer_size) noexcept {
        // 超过block size的直接创建独立节点
        if (unlikely(buffer_size > _block_size)) {
            _blocks.emplace_front(buffer_size);
            return &_blocks.front()[0];
        }
        // 空间不足时扩展一个新的block
        if (unlikely(!block_sufficient(buffer_size))) {
            expend_block();
        }
        return allocate_in_block(buffer_size);
    }

    inline void expend_block() noexcept {
        _blocks.emplace_back(_block_size);
        _size_in_block = 0;
    }

    inline bool block_sufficient(size_t buffer_size) const noexcept {
        return !_blocks.empty() && _size_in_block + buffer_size <= _block_size;
    }

    inline char* allocate_in_block(size_t buffer_size) noexcept {
        auto& block = _blocks.back();
        char* buffer_start = &block[_size_in_block];
        _size_in_block += buffer_size;
        return buffer_start;
    }

    ::std::list<::std::vector<char>> _blocks;
    size_t _block_size;
    size_t _size_in_block;
};

//template <typename T>
//class ReadOnlyAllocator {
//public:
//    typedef T* pointer;
//    typedef const T* const_pointer;
//    typedef T value_type;

//    ReadOnlyAllocator(ReadOnlyBuffer& buffer) noexcept : _buffer(buffer) {}

//    ReadOnlyAllocator(const ReadOnlyAllocator& other) noexcept :
//        _buffer(other.buffer) {}

//    T* allocate(size_t n) noexcept {
//        return _buffer.create_fixed_buffer(sizeof(T) * n);
//    }

//private:
//    ReadOnlyBuffer& _buffer;
//};

}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif //BAIDU_FEED_MLARCH_BABYLON_BUFFER_H
