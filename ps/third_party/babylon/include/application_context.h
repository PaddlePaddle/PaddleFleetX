#ifndef BAIDU_FEED_MLARCH_BABYLON_APPLICATION_CONTEXT_H
#define BAIDU_FEED_MLARCH_BABYLON_APPLICATION_CONTEXT_H

#include <memory>
#include <unordered_map>
#include <stdlib.h>
#include <inttypes.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "any.h"
#include "expect.h"
#include "string_util.h"

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

// IOC容器，实现组件的注册和获取
// 组件可以是任意类型，且不需要公共基类
// 支持组件组装，和宏支持下的自动组装
class ApplicationContext {
public:
    // 单一组件的容器，包装了对具体组件进行初始化和组装的方法
    // 在context initialize时，会initialize并wireup每一个组件
    class ComponentHolder {
    public:
        virtual ~ComponentHolder() noexcept {}
        // initialize之前无法根据实际对象确定类型
        // 留一个接口用于问题定位日志，没有其他作用
        virtual const ::std::string& type() const noexcept;
        // 初始化函数，构建一个组件，并调用set设置到容器中
        virtual int32_t initialize() noexcept;
        // 组装一个组件，可以从context获取其他组件并记录其指针
        virtual int32_t wireup(ApplicationContext&) noexcept;

    protected:
        // 检测T是否具有int wireup(ApplicationContext&)函数
        template <typename T>
        class Wireable {
        public:
            template <typename U = T>
            static auto checker(int32_t) -> typename ::std::conditional<
                ::std::is_same<int32_t,
                    decltype(
                        ::std::declval<U>().wireup(::std::declval<
                            ApplicationContext&>()))
                >::value, std::true_type, ::std::false_type>::type;
            template <typename U = T>
            static ::std::false_type checker(...);

            static const bool value = ::std::is_same<decltype(checker<T>(0)), ::std::true_type>::value;
        };

        // 检测T是否具有int initialize()函数
        template <typename T>
        class Initializeable {
        public:
            template <typename U = T>
            static auto checker(int32_t) -> typename ::std::conditional<
                ::std::is_same<int32_t,
                    decltype(
                        ::std::declval<U>().initialize())
                >::value, std::true_type, ::std::false_type>::type;
            template <typename U = T>
            static ::std::false_type checker(...);

            static const bool value = ::std::is_same<decltype(checker<T>(0)), ::std::true_type>::value;
        };

        // 设置初始化好的组件到容器中
        template <typename T>
        inline void set(::std::unique_ptr<T>&& component) noexcept;
        // 设置初始化好的组件到容器中
        template <typename T>
        inline void ref(T& component) noexcept;
        // 取得已经设置到容器中的组件
        template <typename T>
        inline T* get() noexcept;

        template <typename V, typename ::std::enable_if<Wireable<V>::value, int>::type = 0>
        inline int32_t wireup_if_possible(V* component, ApplicationContext& context) noexcept {
            return component->wireup(context);
        }

        template <typename V, typename ::std::enable_if<!Wireable<V>::value, int>::type = 0>
        inline int32_t wireup_if_possible(V*, ApplicationContext&) noexcept {
            return 0;
        }

        template <typename V, typename ::std::enable_if<Initializeable<V>::value, int>::type = 0>
        inline int32_t initialize_if_possible(V* component) noexcept {
            return component->initialize();
        }

        template <typename V, typename ::std::enable_if<!Initializeable<V>::value, int>::type = 0>
        inline int32_t initialize_if_possible(V*) noexcept {
            return 0;
        }
        
    private:
        // 组件的实际类型
        inline const Any::Id& component_type() const noexcept;
        // 组件的名字，在注册时设置，由context使用
        inline ::std::string& name() noexcept;

        Any _component;
        ::std::string _name;

        friend class ApplicationContext;
    };

    // 单例context
    static ApplicationContext& instance() noexcept {
        static ApplicationContext _instance;
        return _instance;
    }

    ApplicationContext() = default;
    ApplicationContext(const ApplicationContext&) = delete;
    ApplicationContext(ApplicationContext&& other) noexcept :
        _holders(::std::move(other._holders)),
        _holder_by_type(::std::move(other._holder_by_type)),
        _holder_by_name(::std::move(other._holder_by_name)) {}

    // 注册一个组件容器，默认没有名字
    void register_component(::std::unique_ptr<ComponentHolder>&& holder,
        const ::std::string& name = "") noexcept {
        holder->name() = name;
        _holders.emplace_back(::std::move(holder));
    }

    template <typename H>
    void register_component(H&& holder, const ::std::string& name = "") noexcept {
        register_component(::std::unique_ptr<ComponentHolder>(
                new H(::std::move(holder))), name);
    }

    // 初始化整个context，对注册过的所有组件进行初始化和组装
    int32_t initialize() noexcept {
        for (auto& holder : _holders) {
            // 依次initialize
            LOG(INFO) << "initializing component[" << holder->name() << "] of type "
                << holder->type();
            if (0 != holder->initialize()) {
                LOG(WARNING) << "initialize failed component[" << holder->name() << "] of type "
                    << holder->type(); 
                return -1;
            }
            LOG(INFO) << "initialize successful component[" << holder->name() << "] of type "
                << holder->component_type().name;
            // 创建type和type-name的映射
            if (0 != map_component(holder.get())) {
                return -2;
            }
        }
        for (auto& holder : _holders) {
            // 依次wireup
            LOG(INFO) << "wiring up component[" << holder->name() << "] of type "
                << holder->component_type().name;
            if (0 != holder->wireup(*this)) {
                LOG(WARNING) << "wireup failed component[" << holder->name() << "] of type "
                    << holder->component_type().name; 
                return -1;
            }
            LOG(INFO) << "wireup successful component[" << holder->name() << "] of type "
                << holder->component_type().name;
        }
        return 0;
    }

    // 按类型获取组件
    template <typename T>
    inline T* get() noexcept {
        typedef typename ::std::remove_cv<T>::type TT;
        auto it = _holder_by_type.find(&Any::TypeId<TT>::ID);
        if (it != _holder_by_type.end() && it->second != nullptr) {
            return it->second->get<TT>();
        }
        return nullptr;
    }

    // 按类型获取组件，推导版本
    template <typename T>
    inline int32_t get(T*& component) noexcept {
        component = get<T>();
        return component != nullptr ? 0 : -1;
    }

    // 按类型+名字获取组件
    template <typename T>
    inline T* get(const ::std::string& name) noexcept {
        thread_local ::std::string real_name;
        build_real_name<T>(real_name, name);
        auto it = _holder_by_name.find(real_name);
        if (it != _holder_by_name.end()) {
            return it->second->get<T>();
        }
        return nullptr;
    }

    // 按类型+名字获取组件，推导版本
    template <typename T>
    inline int32_t get(T*& component, const ::std::string& name) noexcept {
        component = get<T>(name);
        return component != nullptr ? 0 : -1;
    }

private:
    // 创建组件映射
    inline int32_t map_component(ComponentHolder* holder) noexcept {
        // 创建type映射
        {
            auto result = _holder_by_type.emplace(
                ::std::make_pair(&holder->component_type(), holder));
            // type冲突，设置无法只按type获取
            if (result.second == false) {
                result.first->second = nullptr;
            }
        }
        // 创建type + name映射
        {
            thread_local ::std::string real_name;
            build_real_name(real_name, holder->name(), holder->component_type());
            auto result = _holder_by_name.emplace(
                ::std::make_pair(real_name, holder));
            if (result.second == false) {
                LOG(WARNING) << "both name and type conflict component[" << holder->name() << "] of type "
                    << holder->component_type().name;
                return -2;
            }
        }
        return 0;
    }

    // 创建type + name签名，给出TypeId
    inline static void build_real_name(::std::string& real_name,
        const ::std::string& name, const Any::Id& type) noexcept {
        strprintf(real_name, name.size() + 32, "%p\t%s", &type, name.c_str());
    }

    // 创建type + name签名，从模板T推导TypeId
    template <typename T>
    inline static void build_real_name(::std::string& real_name,
        const ::std::string& name) noexcept {
        build_real_name(real_name, name, Any::TypeId<T>::ID);
    }

    ::std::vector<::std::unique_ptr<ComponentHolder>> _holders;
    ::std::unordered_map<const Any::Id*, ComponentHolder*> _holder_by_type;
    ::std::unordered_map<::std::string, ComponentHolder*> _holder_by_name;
};

template <typename T>
void ApplicationContext::ComponentHolder::set(::std::unique_ptr<T>&& component) noexcept {
    _component = ::std::move(component);
}

template <typename T>
void ApplicationContext::ComponentHolder::ref(T& component) noexcept {
    _component.ref(component);
}

template <typename T>
T* ApplicationContext::ComponentHolder::get() noexcept {
    return _component.get<T>();
}

const Any::Id& ApplicationContext::ComponentHolder::component_type() const noexcept {
    return _component.instance_type();
}

::std::string& ApplicationContext::ComponentHolder::name() noexcept {
    return _name;;
}

// 辅助自动组装的宏
#define __BABYLON_DECLARE_AUTOWIRE_MEMBER(type, member, name) type member;
#define __BABYLON_AUTOWIRE_MEMBER(type, member, name) \
    ret += context.get(member); \
    if (member == nullptr) { \
        LOG(WARNING) << "get component[]" \
            << " with type[" << #type << "] failed"; \
    }
#define __BABYLON_AUTOWIRE_MEMBER_BY_NAME(type, member, name) \
    ret += context.get(member, #name); \
    if (member == nullptr) { \
        LOG(WARNING) << "get component[" << #name << "]" \
            << " with type[" << #type << "] failed"; \
    }
#define __BABYLON_DECLARE_EACH_AUTOWIRE_MEMBER(r, data, args) \
    BOOST_PP_EXPR_IIF(1, \
    __BABYLON_DECLARE_AUTOWIRE_MEMBER BOOST_PP_SEQ_ELEM(1, args))
#define __BABYLON_AUTOWIRE_EACH_MEMBER(r, data, args) \
    BOOST_PP_EXPR_IIF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, args)), 0), \
        __BABYLON_AUTOWIRE_MEMBER BOOST_PP_SEQ_ELEM(1, args)) \
    BOOST_PP_EXPR_IIF(BOOST_PP_EQUAL(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, args)), 1), \
        __BABYLON_AUTOWIRE_MEMBER_BY_NAME BOOST_PP_SEQ_ELEM(1, args))
// 自动组装一个成员，按类型从context获取
#define BABYLON_MEMBER(type, member) (((0))((type, member, _)))
// 自动组装一个成员，按类型+名字从context获取
#define BABYLON_MEMBER_BY_NAME(type, member, name) (((1))((type, member, name)))
// 生成自动组装函数和成员，例如
//struct AutowireComponent {
//    // 生成自动组装
//    BABYLON_AUTOWIRE(
//        // 生成成员
//        // NormalComponent* nc;
//        // 并组装为nc = context.get<NormalComponent>();
//        BABYLON_MEMBER(NormalComponent*, nc)
//        BABYLON_MEMBER(const NormalComponent*, cnc)
//        // 生成成员
//        // AnotherNormalComponent* anc1;
//        // 并组装为anc1 = context.get<AnotherNormalComponent>("name1");
//        BABYLON_MEMBER_BY_NAME(AnotherNormalComponent*, anc1, name1)
//        BABYLON_MEMBER_BY_NAME(AnotherNormalComponent*, anc2, name2)
//    )
//};
#define BABYLON_AUTOWIRE(members) \
    int32_t wireup(::baidu::feed::mlarch::babylon::ApplicationContext& context) noexcept { \
        int32_t ret = 0; \
        BOOST_PP_SEQ_FOR_EACH(__BABYLON_AUTOWIRE_EACH_MEMBER, _, members) \
        return ret; \
    } \
    BOOST_PP_SEQ_FOR_EACH(__BABYLON_DECLARE_EACH_AUTOWIRE_MEMBER, _, members) \
    friend class ::baidu::feed::mlarch::babylon::ApplicationContext::ComponentHolder;


}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif //BAIDU_FEED_MLARCH_BABYLON_APPLICATION_CONTEXT_H

#include "component_holder.h"
