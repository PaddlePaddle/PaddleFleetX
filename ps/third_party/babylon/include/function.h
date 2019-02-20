#ifndef BAIDU_FEED_MLARCH_BABYLON_FUNCTION_H
#define BAIDU_FEED_MLARCH_BABYLON_FUNCTION_H

#include <functional>
#include <stdlib.h>
#include <inttypes.h>
#include <base/logging.h>

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

// c++11规范下，::std::function要求包装目标必须可以拷贝构造
// 即使业务自身保证不会发生拷贝
// 此时可以用MoveableFunction进行支持，但是需要业务确保【不会拷贝】
template <typename C>
class MoveableFunction {
public:
    // 移动构造包装目标
    inline MoveableFunction(C&& callback) noexcept :
        _callback(::std::move(callback)) {}
    // 移动是移动目标
    inline MoveableFunction(MoveableFunction&& other) noexcept :
        _callback(::std::move(other._callback)) {}
    // 伪造拷贝构造函数，只用于欺骗::std::function
    // 如果拷贝真实发生，会强制终止运行
    inline MoveableFunction(const MoveableFunction& other) noexcept :
        _callback(::std::move(const_cast<C&>(other._callback))) {
        LOG(FATAL) << "unexpected copy happen, maybe a bug?";
        abort();
    }
    // 代理invoke持有的目标
    template <typename ...A>
    inline decltype(::std::declval<C>()(::std::declval<A>()...)) operator() (A... args) noexcept {
        return _callback(::std::forward<A>(args)...);
    }

private:
    C _callback;
};

// 包装可以被::std::function持有的目标
// 对于可以拷贝构造的目标，直接移动返回目标本身
template <typename C,
         typename ::std::enable_if<::std::is_copy_constructible<C>::value, int32_t>::type = 0>
C wrap_moveable_function(C&& c) {
    return ::std::move(c);
}

// 包装可以被::std::function持有的目标
// 对于不可拷贝构造的目标，包装成MoveableFunction欺骗::std::function
template <typename C,
         typename ::std::enable_if<!::std::is_copy_constructible<C>::value, int32_t>::type = 0>
MoveableFunction<C> wrap_moveable_function(C&& c) {
    return MoveableFunction<C>(::std::move(c));
}

}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif //BAIDU_FEED_MLARCH_BABYLON_FUNCTION_H
