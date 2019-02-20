#ifndef BAIDU_FEED_MLARCH_BABYLON_ANY_H
#define BAIDU_FEED_MLARCH_BABYLON_ANY_H

#include <string>
#include <memory>
#include <glog/logging.h>

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

// 类似boost::any可以用于存放任何对象，并支持按原类型取回
// 相比boost::any支持不可拷贝，不可移动构造的对象
// 并支持引用构造，可以支持同构处理引用和实体any对象
class Any {
public:
    template<class T>
    class TypeId;

    // 用于标记一个类型的唯一标记，可以从name取到可读表达
    // 只能通过TypeId<T>获取，不可保存复制
    // 由TypeId确保同T返回同对象，所以相等通过指针相等判断
    class Id {
    public:
        const ::std::string name;

        inline bool operator==(const Id& other) const noexcept {
            return this == &other;
        }

    private:
        Id(const ::std::string& type_name) noexcept : name(type_name) {}
        Id(const Id&) = delete;
        Id(Id&&) = delete;

        template <typename T>
        friend class TypeId;
    };

    // 获取类型T的唯一标记
    template<class T>
    class TypeId {
    public:
        static const Id ID;

    private:
        static const char* name() noexcept {
            return __PRETTY_FUNCTION__;
        }
    };

    enum class Type : uint8_t {
        INSTANCE = 0,
        INT64,
        INT32,
        INT16,
        INT8,
        UINT64,
        UINT32,
        UINT16,
        UINT8,
        BOOLEAN
    };

    enum class ReferenceType : uint8_t {
        NOT_REFERENCE = 0,
        CONST_REFERENCE,
        MUTABLE_REFERENCE
    };

    // 默认构造，后续可用 = 赋值
    inline Any() noexcept = default;

    // 拷贝构造，如果other中已经持有一个对象实体
    // 需要确保具有拷贝构造能力，否则会强制abort退出
    // 如果other中持有对象引用，则会构造一个相同的对象引用
    inline Any(const Any& other) noexcept :
        _empty(other._empty),
        _type(other._type),
        _reference_type(other._reference_type),
        _type_id(other._type_id),
        _primitive_holder(other._primitive_holder),
        _instance_holder(other._instance_holder ? other._instance_holder->copy() : nullptr),
        _pointer(_instance_holder ? _instance_holder->get() :
            (_reference_type == ReferenceType::NOT_REFERENCE ?
                reinterpret_cast<const void*>(&_primitive_holder) : other._pointer)) {}

    // 拷贝构造的非const参数版本为了避免被T&&捕获
    inline Any(Any& other) noexcept : Any(const_cast<const Any&>(other)) {}

    // 移动构造函数，移动other到自身，不会构造对象本体
    inline Any(Any&& other) noexcept :
        _empty(other._empty),
        _type(other._type),
        _reference_type(other._reference_type),
        _type_id(other._type_id),
        _primitive_holder(other._primitive_holder),
        _instance_holder(::std::move(other._instance_holder)),
        _pointer(_type == Type::INSTANCE ? other._pointer :
                reinterpret_cast<const void*>(&_primitive_holder)) {
        other.clear();
    }

    // 从对象指针移动构造，可支持不能移动构造的对象
    template<typename T>
    inline Any(::std::unique_ptr<T>&& value) noexcept :
        _empty(false),
        _type(Type::INSTANCE),
        _reference_type(ReferenceType::NOT_REFERENCE),
        _type_id(&TypeId<T>::ID),
        _primitive_holder(0),
        _instance_holder(new InstanseHolter<T>(::std::move(value))),
        _pointer(_instance_holder->get()) {}

    // 从对象拷贝或移动构造，自身会持有对象本体的拷贝或移动构造副本
    template<typename T>
    inline Any(T&& value) noexcept : Any(::std::unique_ptr<
        typename ::std::remove_cv<typename ::std::remove_reference<T>::type>::type>
            (new typename ::std::remove_cv<typename ::std::remove_reference<T>::type>::type(
                ::std::forward<T>(value)))) {}

    // 对整数特化支持
    inline Any(int64_t value) noexcept :
        Any(value, Type::INT64, TypeId<int64_t>::ID) {}
    inline Any(int32_t value) noexcept :
        Any(value, Type::INT32, TypeId<int32_t>::ID) {}
    inline Any(int16_t value) noexcept :
        Any(value, Type::INT16, TypeId<int16_t>::ID) {}
    inline Any(int8_t value) noexcept :
        Any(value, Type::INT8, TypeId<int8_t>::ID) {}
    inline Any(uint64_t value) noexcept :
        Any(value, Type::UINT64, TypeId<uint64_t>::ID) {}
    inline Any(uint32_t value) noexcept :
        Any(value, Type::UINT32, TypeId<uint32_t>::ID) {}
    inline Any(uint16_t value) noexcept :
        Any(value, Type::UINT16, TypeId<uint16_t>::ID) {}
    inline Any(uint8_t value) noexcept :
        Any(value, Type::UINT8, TypeId<uint8_t>::ID) {}
    inline Any(bool value) noexcept :
        Any(value, Type::BOOLEAN, TypeId<bool>::ID) {}

    // 类似拷贝构造的拷贝赋值
    inline Any& operator=(const Any& other) noexcept {
        _empty = other._empty;
        _type = other._type;
        _reference_type = other._reference_type;
        _type_id = other._type_id;
        _primitive_holder = other._primitive_holder;
        _instance_holder.reset(other._instance_holder ? other._instance_holder->copy() : nullptr);
        _pointer = _instance_holder ? _instance_holder.get() :
            (_reference_type == ReferenceType::NOT_REFERENCE ?
             reinterpret_cast<const void*>(&_primitive_holder) : other._pointer);
        return *this;
    }

    // 类似移动构造的移动赋值
    inline Any& operator=(Any&& other) noexcept {
        _empty = other._empty;
        _type = other._type;
        _reference_type = other._reference_type;
        _type_id = other._type_id;
        _primitive_holder = other._primitive_holder;
        _instance_holder = ::std::move(other._instance_holder);
        _pointer = _type == Type::INSTANCE ? other._pointer :
             reinterpret_cast<const void*>(&_primitive_holder);
        other.clear();
        return *this;
    }

    // 类似对象拷贝或移动构造的赋值
    template<typename T>
    inline Any& operator=(T&& value) noexcept {
        typedef typename ::std::remove_cv<
            typename ::std::remove_reference<T>::type>::type RealType;
        return operator=(::std::unique_ptr<RealType>(new RealType(::std::forward<T>(value))));
    }

    template<typename T>
    inline Any& operator=(::std::unique_ptr<T>&& value) noexcept {
        _empty = false;
        _type = Type::INSTANCE;
        _reference_type = ReferenceType::NOT_REFERENCE;
        _type_id = &TypeId<T>::ID;
        _primitive_holder = 0;
        _instance_holder.reset(new InstanseHolter<T>(::std::move(value)));
        _pointer = _instance_holder->get();
        return *this;
    }

    // 对整数特化支持的赋值
    inline Any& operator=(int64_t value) noexcept {
        return assign(value, Type::INT64, TypeId<int64_t>::ID);
    }
    inline Any& operator=(int32_t value) noexcept {
        return assign(value, Type::INT32, TypeId<int32_t>::ID);
    }
    inline Any& operator=(int16_t value) noexcept {
        return assign(value, Type::INT16, TypeId<int16_t>::ID);
    }
    inline Any& operator=(int8_t value) noexcept {
        return assign(value, Type::INT8, TypeId<int8_t>::ID);
    }
    inline Any& operator=(uint64_t value) noexcept {
        return assign(value, Type::UINT64, TypeId<uint64_t>::ID);
    }
    inline Any& operator=(uint32_t value) noexcept {
        return assign(value, Type::UINT32, TypeId<uint32_t>::ID);
    }
    inline Any& operator=(uint16_t value) noexcept {
        return assign(value, Type::UINT16, TypeId<uint16_t>::ID);
    }
    inline Any& operator=(uint8_t value) noexcept {
        return assign(value, Type::UINT8, TypeId<uint8_t>::ID);
    }
    inline Any& operator=(bool value) noexcept {
        return assign(value, Type::BOOLEAN, TypeId<bool>::ID);
    }

    // 引用另一个Any，可以像被引用的Any一样使用使用
    // 并保留const属性（假如被引用的Any时const引用，那么也只能const get）
    // 读写目标都是被引用的Any中的数据
    inline Any& ref(Any& value) noexcept {
        _empty = value._empty;
        _type = value._type;
        _reference_type = value._reference_type == ReferenceType::CONST_REFERENCE ?
            ReferenceType::CONST_REFERENCE : ReferenceType::MUTABLE_REFERENCE;
        _type_id = &value.instance_type();
        _primitive_holder = 0;
        _instance_holder.reset();
        _pointer = value._pointer;
        return *this;
    }

    // const版本
    inline Any& ref(const Any& value) noexcept {
        return cref(value);
    }

    // 方便显示调用的const版本
    inline Any& cref(const Any& value) noexcept {
        _empty = value._empty;
        _type = value._type;
        _reference_type = ReferenceType::CONST_REFERENCE;
        _type_id = &value.instance_type();
        _primitive_holder = 0;
        _instance_holder.reset();
        _pointer = value._pointer;
        return *this;
    }

    // 引用对象，本身不会进行计数等生命周期维护
    // 可以像持有对象本体一样使用const get取值
    template<typename T>
    inline Any& ref(const T& value) noexcept {
        return cref(value);
    }

    template<typename T>
    inline Any& cref(const T& value) noexcept {
        _empty = false;
        _type = Type::INSTANCE;
        _reference_type = ReferenceType::CONST_REFERENCE;
        _type_id = &TypeId<T>::ID;
        _primitive_holder = 0;
        _instance_holder.reset();
        _pointer = &value;
        return *this;
    }

    // 引用对象的非常量版，支持const/非const get取值
    template<typename T>
    inline Any& ref(T& value) noexcept {
        _empty = false;
        _type = Type::INSTANCE;
        _reference_type = ReferenceType::MUTABLE_REFERENCE;
        _type_id = &TypeId<T>::ID;
        _primitive_holder = 0;
        _instance_holder.reset();
        _pointer = &value;
        return *this;
    }

    // 重置，释放持有的本体，恢复到empty状态
    inline void clear() noexcept {
        _empty = true;
        _type = Type::INSTANCE;
        _reference_type = ReferenceType::NOT_REFERENCE;
        _type_id = &TypeId<void>::ID;
        _primitive_holder = 0;
        _instance_holder.reset();
        _pointer = nullptr;
    }

    // 取值，需要指定和存入时一致的类型才可取出
    // empty或者类型不对都会取到nullptr
    // 尝试获取const ref也会取到nullptr
    template<typename T>
    inline T* get() noexcept {
        if (_reference_type != ReferenceType::CONST_REFERENCE
            && _type_id == &TypeId<T>::ID) {
            return reinterpret_cast<T*>(const_cast<void*>(_pointer));
        }
        return nullptr;
    }

    // 常量取值
    template<typename T>
    inline const T* cget() const noexcept {
        if (_type_id == &TypeId<T>::ID) {
            return reinterpret_cast<const T*>(_pointer);
        }
        return nullptr;
    }

    // 常量取值
    template<typename T>
    inline const T* get() const noexcept {
        return cget<T>();
    }

    // 辅助判断是否是常量引用
    inline bool is_const_reference() const {
        return _reference_type == ReferenceType::CONST_REFERENCE;
    }

    // 辅助判断是否是引用
    inline bool is_reference() const {
        return _reference_type != ReferenceType::NOT_REFERENCE;
    }

    // integer支持的转化功能
    template<typename T>
    inline T as() const noexcept {
        return *reinterpret_cast<const int64_t*>(_pointer);
    }

    // 辅助判断是否为空
    inline operator bool() const noexcept {
        return !_empty;
    }

    // 辅助判断类型
    inline Type type() const noexcept {
        return _type;
    }

    // 辅助判断对象类型，可通过字符串一致判断同一类型
    // 也可通过同地址判断同一类型
    // 可以产生人可读的类型表达
    inline const Id& instance_type() const noexcept {
        return *_type_id;
    }

private:
    class Holder {
    public:
        virtual ~Holder() noexcept {};
        virtual void* get() noexcept = 0;
        virtual Holder* copy() const noexcept = 0;
        virtual bool copyable() const noexcept = 0;
    };

    template<typename T>
    class InstanseHolter : public Holder {
    public:
        InstanseHolter(::std::unique_ptr<T>&& instance) noexcept :
            _instanse(::std::move(instance)) {
        }

        virtual void* get() noexcept override {
            return _instanse.get();
        }

        virtual bool copyable() const noexcept override {
            return ::std::is_copy_constructible<T>::value;
        }

        virtual Holder* copy() const noexcept override {
            return do_copy();
        }

    private:
        template <typename U = T, typename ::std::enable_if<::std::is_copy_constructible<U>::value, int>::type = 0>
        inline Holder* do_copy() const noexcept override {
            ::std::unique_ptr<T> copyed_instance(
                new T(*_instanse.get()));
            return new InstanseHolter(::std::move(copyed_instance));
        }

        template <typename U = T, typename ::std::enable_if<!::std::is_copy_constructible<U>::value, int>::type = 0>
        inline Holder* do_copy() const noexcept override {
            LOG(FATAL) << "try copy non-copyable instance by copy an babylon::Any instance";
            abort();
        }

    private:
        ::std::unique_ptr<T> _instanse;
    };

    inline Any(int64_t value, Type type, const Id& instance_type) noexcept :
        _empty(false),
        _type(type),
        _reference_type(ReferenceType::NOT_REFERENCE),
        _type_id(&instance_type),
        _primitive_holder(value),
        _pointer(&_primitive_holder) {}

    inline Any& assign(int64_t value, Type type, const Id& instance_type) noexcept {
        _empty = false;
        _type = type;
        _reference_type = ReferenceType::NOT_REFERENCE;
        _type_id = &instance_type;
        _primitive_holder = value;
        _instance_holder.reset();
        _pointer = &_primitive_holder;
        return *this;
    }

    bool _empty {true};
    Type _type {Type::INSTANCE};
    ReferenceType _reference_type {ReferenceType::NOT_REFERENCE};
    const Id* _type_id {&TypeId<void>::ID};
    int64_t _primitive_holder {0};
    ::std::unique_ptr<Holder> _instance_holder {nullptr};
    const void* _pointer {nullptr};
};

template<class T>
const Any::Id Any::TypeId<T>::ID(Any::TypeId<T>::name());

} // babylon
} // mlarch
} // feed
} // baidu

#endif // BAIDU_FEED_MLARCH_BABYLON_ANY_H
