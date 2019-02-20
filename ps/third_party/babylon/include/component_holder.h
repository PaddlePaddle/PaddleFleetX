#ifndef BAIDU_FEED_MLARCH_BABYLON_COMPONENT_HOLDER_H
#define BAIDU_FEED_MLARCH_BABYLON_COMPONENT_HOLDER_H

#include <inttypes.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "any.h"
#include "expect.h"
#include "application_context.h"

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

// 应用于某个具体类型的组件容器
template <typename T>
class TypedComponentHolder : public ApplicationContext::ComponentHolder {
public:
    virtual const ::std::string& type() const noexcept;
};

// 支持自动装配的组件容器
// 探测装配函数，并调用
template <typename T, typename U = T>
class AutowireComponentHolder : public TypedComponentHolder<T> {
public:
    virtual int32_t wireup(ApplicationContext& context) noexcept;
};

// 默认组件容器模板，使用默认构造函数创建组件
// 探测初始化函数和组装函数，并调用
template <typename T, typename U = T>
class DefaultComponentHolder : public AutowireComponentHolder<T, U> {
public:
    virtual int32_t initialize() noexcept;
};

// 自动使用默认组件容器模板生成一个ComponentHolder，并注册到context
// 可以为组件指定一个name，以及注册到那个context
// 默认没有名字，注册到全局单例的context
template <typename T, typename U = T>
class DefaultComponentRegister {
public:
    DefaultComponentRegister(const ::std::string& name = "",
        ApplicationContext& context = ApplicationContext::instance()) noexcept;
};

// 外部初始化的组件容器模板，使用已经创建并初始化好的实例创建组件
// 探测组装函数，并调用
template <typename T, typename U = T>
class InitializedComponentHolder : public AutowireComponentHolder<T, U> {
public:
    InitializedComponentHolder(::std::unique_ptr<T>&& component) noexcept;
    InitializedComponentHolder(T&& component) noexcept;
    InitializedComponentHolder(T& component) noexcept;
};

// 自定义组件容器，使用自定义函数创建并初始化组件
// 探测装配函数，并调用
template <typename T>
class CustomComponentHolder : public AutowireComponentHolder<T, T> {
public:
    CustomComponentHolder(::std::function<T*()>&& creator) noexcept;
    virtual int32_t initialize() noexcept;
private:
    ::std::function<T*()> _creator;
};

// 自动使用自定义组件容器模板生成一个ComponentHolder，并注册到context
// 可以为组件指定一个name，以及注册到那个context
// 默认没有名字，注册到全局单例的context
template <typename T>
class CustomComponentRegister {
public:
    CustomComponentRegister(::std::function<T*()>&& creator, const ::std::string& name = "",
        ApplicationContext& context = ApplicationContext::instance()) noexcept;
};

///////////////////////////////////////////////////////////////////////////////
// TypedComponentHolder begin
template <typename T>
const ::std::string& TypedComponentHolder<T>::type() const noexcept {
    return Any::TypeId<T>::ID.name;
}
// TypedComponentHolder end
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// AutowireComponentHolder begin
template <typename T, typename U>
int32_t AutowireComponentHolder<T, U>::wireup(ApplicationContext& context) noexcept {
    return this->wireup_if_possible(static_cast<T*>(
            this->template get<U>()), context);
}
// AutowireComponentHolder end
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// DefaultComponentHolder begin
template <typename T, typename U>
int32_t DefaultComponentHolder<T, U>::initialize() noexcept {
    auto component_pointer = new T();
    ::std::unique_ptr<U> component(static_cast<U*>(component_pointer));
    if (0 != this->initialize_if_possible(component_pointer)) {
        return -1;
    }
    this->set(::std::move(component));
    return 0;
}
// DefaultComponentHolder end
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// DefaultComponentRegister begin
template <typename T, typename U>
DefaultComponentRegister<T, U>::DefaultComponentRegister(const ::std::string& name,
    ApplicationContext& context) noexcept {
    context.register_component(DefaultComponentHolder<T, U>(), name);
}
// DefaultComponentRegister end
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// InitializedComponentHolder begin
template <typename T, typename U>
InitializedComponentHolder<T, U>::InitializedComponentHolder(
    ::std::unique_ptr<T>&& component) noexcept {
    this->set(::std::unique_ptr<U>(::std::move(component)));
}

template <typename T, typename U>
InitializedComponentHolder<T, U>::InitializedComponentHolder(T&& component) noexcept :
    InitializedComponentHolder<T, U>(::std::unique_ptr<T>(new T(::std::move(component)))) {}

template <typename T, typename U>
InitializedComponentHolder<T, U>::InitializedComponentHolder(T& component) noexcept {
    this->ref(component);
}
// InitializedComponentHolder end
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// CustomComponentHolder begin
template <typename T>
CustomComponentHolder<T>::CustomComponentHolder(
    ::std::function<T*()>&& creator) noexcept {
    _creator = ::std::move(creator);
}

template <typename T>
int32_t CustomComponentHolder<T>::initialize() noexcept {
    auto component = _creator();
    if (unlikely(component == nullptr)) {
        LOG(WARNING) << "initialize custom component of type " << this->type() << " failed";
        return -1;
    }
    this->set(::std::unique_ptr<T>(component));
    return 0;
}
// CustomComponentHolder end
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// DefaultComponentRegister begin
template <typename T>
CustomComponentRegister<T>::CustomComponentRegister(::std::function<T*()>&& creator,
    const ::std::string& name, ApplicationContext& context) noexcept {
    context.register_component(CustomComponentHolder<T>(::std::move(creator)), name);
}
// DefaultComponentRegister end
///////////////////////////////////////////////////////////////////////////////

// 用名字name注册T类型的组件
#define BABYLON_REGISTER_COMPONENT_WITH_NAME(T, name) \
    static ::baidu::feed::mlarch::babylon::DefaultComponentRegister<T> \
        BOOST_PP_CAT(Babylon_Application_Context_Register, __COUNTER__)(#name);

// 用匿名方式注册T类型的组件
// T为一个标准名，如果不满足，如包含模板参数等，需前置一个typedef
#define BABYLON_REGISTER_COMPONENT(T) \
    static ::baidu::feed::mlarch::babylon::DefaultComponentRegister<T> \
        BOOST_PP_CAT(Babylon_Application_Context_Register, __COUNTER__);

// 用名字name注册实际T类型，暴露U类型的组件
#define BABYLON_REGISTER_COMPONENT_WITH_TYPE_NAME(T, U, name) \
    static ::baidu::feed::mlarch::babylon::DefaultComponentRegister<T, U> \
        BOOST_PP_CAT(Babylon_Application_Context_Register, __COUNTER__)(#name);

// 用名字name注册T类型的组件
#define BABYLON_REGISTER_CUSTOM_COMPONENT_WITH_NAME(T, name, ...) \
    static ::baidu::feed::mlarch::babylon::CustomComponentRegister<T> \
        BOOST_PP_CAT(Babylon_Application_Context_Register, __COUNTER__)(__VA_ARGS__, #name);

// 用匿名方式注册T类型的组件
// T为一个标准名，如果不满足，如包含模板参数等，需前置一个typedef
#define BABYLON_REGISTER_CUSTOM_COMPONENT(T, ...) \
    static ::baidu::feed::mlarch::babylon::CustomComponentRegister<T> \
        BOOST_PP_CAT(Babylon_Application_Context_Register, __COUNTER__)(__VA_ARGS__);

}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif //BAIDU_FEED_MLARCH_BABYLON_COMPONENT_HOLDER_H
