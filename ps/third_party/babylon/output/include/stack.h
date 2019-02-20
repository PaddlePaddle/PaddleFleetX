#ifndef BAIDU_FEED_MLARCH_BABYLON_STACK_H
#define BAIDU_FEED_MLARCH_BABYLON_STACK_H

#include <cstddef>
#include <utility>
#include <baidu/feed/mlarch/babylon/expect.h>

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

template<typename T>
class Stack {
public:
    inline Stack(T* array, size_t capacity) noexcept :
        _array(array), _capacity(capacity) {}

    inline bool empty() const noexcept {
        return _back == -1;
    }
    
    inline bool emplace_back(T&& value) noexcept {
        int32_t back = _back + 1;
        if (unlikely(back >= _capacity)) {
            return false;
        }
        
        _array[back] = ::std::move(value);
        _back = back;
        return true;
    }

    inline bool emplace_back(const T& value) noexcept {
        return emplace_back(T(value));
    }

    inline bool emplace(T&& value) noexcept {
        return emplace_back(::std::move(value));
    }

    inline bool emplace(const T& value) noexcept {
        return emplace_back(T(value));
    }

    inline T& back() noexcept {
        return _array[_back];
    }

    inline bool pop_back() noexcept {
        if (unlikely(_back < 0)) {
            return false;
        }
        _back--;
        return true;
    }

    inline bool pop() noexcept {
        return pop_back();
    }

    inline size_t size() const noexcept {
        return _back + 1;
    }

    inline size_t capacity() const noexcept {
        return _capacity;
    }
    
    inline void clear() noexcept {
        _back = -1;
    }

    inline T& operator[](size_t index) noexcept {
        return _array[index];
    }

private:
    T* _array;
    size_t _capacity;
    int32_t _back {-1};
};

#define BABYLON_STACK(type, name, capacity) \
    type __BABYLON_INVISIBLE_VARIABLE_##name[capacity]; \
    ::baidu::feed::mlarch::babylon::Stack<type> name( \
        __BABYLON_INVISIBLE_VARIABLE_##name, \
        sizeof(__BABYLON_INVISIBLE_VARIABLE_##name) / sizeof(type));

}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif //BAIDU_FEED_MLARCH_BABYLON_STACK_H
