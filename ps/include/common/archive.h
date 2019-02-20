#ifndef BAIDU_BAIDU_PSLIB_INCLUDE_COMMON_ARCHIVE_H
#define BAIDU_BAIDU_PSLIB_INCLUDE_COMMON_ARCHIVE_H 

#include <memory>
#include <vector>
#include <map>
#include <set>
#include <valarray>
#include <unordered_map>
#include <unordered_set>
#include <glog/logging.h>
#include "expect.h"

namespace paddle {
namespace ps {

class ArchiveBase { // not a virtual class
protected:
    ArchiveBase() {
    }
    // Archive is not copyable. But to allow move capture by function objects,
    // check it at runtime rather than at compile time.
    ArchiveBase(const ArchiveBase&) {
        LOG(FATAL) << "Not supported";
    }

    ArchiveBase(ArchiveBase && other);

    ~ArchiveBase() {
        free_buffer();
    }
public:
    ArchiveBase& operator=(const ArchiveBase&) {
        LOG(FATAL) << "Not supported";
        return *this;
    }
    ArchiveBase& operator=(ArchiveBase && other);
    char* buffer() {
        return _buffer;
    }
    void set_read_buffer(char* buffer, size_t length, std::function<void(char*)> && deleter) {
        set_buffer(buffer, length, length, std::move(deleter));
    }
    void set_write_buffer(char* buffer, size_t capacity, std::function<void(char*)> && deleter) {
        set_buffer(buffer, 0, capacity, std::move(deleter));
    }
    void set_buffer(char* buffer, size_t length, size_t capacity,
            std::function<void(char*)> && deleter);
    char* cursor() {
        return _cursor;
    }
    void set_cursor(char* cursor) {
        DCHECK(cursor >= _buffer && cursor <= _finish);
        _cursor = cursor;
    }
    void advance_cursor(size_t offset) {
        DCHECK(offset <= size_t(_finish - _cursor));
        _cursor += offset;
    }
    char* finish() {
        return _finish;
    }
    void set_finish(char* finish) {
        DCHECK(finish >= _cursor && finish <= _limit);
        _finish = finish;
    }
    void advance_finish(size_t offset) {
        DCHECK(offset <= size_t(_limit - _finish));
        _finish += offset;
    }
    char* limit() {
        return _limit;
    }
    size_t position() {
        return _cursor - _buffer;
    }
    size_t length() {
        return _finish - _buffer;
    }
    size_t capacity() {
        return _limit - _buffer;
    }
    bool empty() {
        return _finish == _buffer;
    }
    void reset() {
        free_buffer();
        _buffer = NULL;
        _cursor = NULL;
        _finish = NULL;
        _limit = NULL;
    }
    void clear() {
        _cursor = _buffer;
        _finish = _buffer;
    }
    char* release() {
        char* buf = _buffer;
        _buffer = NULL;
        _cursor = NULL;
        _finish = NULL;
        _deleter = nullptr;
        return buf;
    }
    void resize(size_t newsize) {
        if (unlikely(newsize > capacity())) {
            reserve(std::max(capacity() * 2, newsize));
        }

        _finish = _buffer + newsize;
        _cursor = std::min(_cursor, _finish);
    }
    void reserve(size_t newcap);
    void prepare_read(size_t size) {
        if (unlikely(!(size <= size_t(_finish - _cursor)))) {
            CHECK(size <= size_t(_finish - _cursor));
        }
    }
    void prepare_write(size_t size) {
        if (unlikely(size > size_t(_limit - _finish))) {
            reserve(std::max(capacity() * 2, length() + size));
        }
    }
    void read(void* data, size_t size) {
        if (size > 0) {
            prepare_read(size);
            memcpy(data, _cursor, size);
            advance_cursor(size);
        }
    }
    void read_back(void* data, size_t size) {
        if (size > 0) {
            CHECK(size <= size_t(_finish - _cursor));
            memcpy(data, _finish - size, size);
            _finish -= size;
        }
    }
    void write(const void* data, size_t size) {
        if (size > 0) {
            prepare_write(size);
            memcpy(_finish, data, size);
            advance_finish(size);
        }
    }
    template<class T>
    void get_raw(T& x) {
        prepare_read(sizeof(T));
        memcpy(&x, _cursor, sizeof(T));
        advance_cursor(sizeof(T));
    }
    template<class T>
    T get_raw() {
        T x;
        get_raw<T>(x);
        return x;
    }
    template<class T>
    void put_raw(const T& x) {
        prepare_write(sizeof(T));
        memcpy(_finish, &x, sizeof(T));
        advance_finish(sizeof(T));
    }
protected:
    char* _buffer = NULL;
    char* _cursor = NULL;
    char* _finish = NULL;
    char* _limit = NULL;
    std::function<void(char*)> _deleter = nullptr;

    void free_buffer() {
        if (_deleter) {
            _deleter(_buffer);
        }

        _deleter = nullptr;
    }
};

template<class Type>
class Archive {
};

class BinaryArchiveType {
};

//class TextArchiveType {
//};

typedef Archive<BinaryArchiveType> BinaryArchive;
//typedef Archive<TextArchiveType> TextArchive;

template<>
class Archive<BinaryArchiveType> : public ArchiveBase {
public:
    #define PSLIB_REPEAT(T) \
    BinaryArchive& operator>>(T& x) { \
        get_raw(x); \
        return *this; \
    } \
    BinaryArchive& operator<<(const T& x) { \
        put_raw(x); \
        return *this; \
    }
    // avoid using MIO_REPEAT10, which could produce very long output once compiliation error occurs
    PSLIB_REPEAT(int16_t)
    PSLIB_REPEAT(uint16_t)
    PSLIB_REPEAT(int32_t)
    PSLIB_REPEAT(uint32_t)
    PSLIB_REPEAT(int64_t)
    PSLIB_REPEAT(uint64_t)
    PSLIB_REPEAT(float)
    PSLIB_REPEAT(double)
    PSLIB_REPEAT(signed char)
    PSLIB_REPEAT(unsigned char)
    PSLIB_REPEAT(bool)
    #undef PSLIB_REPEAT
    template<class T>
    T get() {
        T x;
        *this >> x;
        return x;
    }
};


/*
template<>
class Archive<TextArchiveType> : public ArchiveBase {
public:
    #define PSLIB_REPEAT(T) \
    TextArchive& operator>>(T& x) { \
        get_arithmetic(x); \
        return *this; \
    } \
    TextArchive& operator<<(const T& x) { \
        put_arithmetic(x); \
        return *this; \
    }
    PSLIB_REPEAT(int16_t)
    PSLIB_REPEAT(uint16_t)
    PSLIB_REPEAT(int32_t)
    PSLIB_REPEAT(uint32_t)
    PSLIB_REPEAT(int64_t)
    PSLIB_REPEAT(uint64_t)
    PSLIB_REPEAT(float)
    PSLIB_REPEAT(double)
    PSLIB_REPEAT(signed char)
    PSLIB_REPEAT(unsigned char)
    PSLIB_REPEAT(bool)
    #undef PSLIB_REPEAT
    char* next_delim() {
        char* next = cursor();

        while (next < finish() && *next != '\t') {
            next++;
        }

        return next;
    }
    template<class T>
    T get() {
        T x;
        *this >> x;
        return x;
    }
    template<class... ARGS>
    void printf(const char* fmt, ARGS && ... args) {
        size_t temp = limit() - finish();
        int len = snprintf(finish(), temp, fmt, args...);
        CHECK(len >= 0);

        if ((size_t)len >= temp) {
            prepare_write(len + 1);
            CHECK(snprintf(finish(), (size_t)len + 1, fmt, args...) == len);
        }

        advance_finish(len);
    }
private:
    template<class T>
    void get_arithmetic(T& x);
    //template<class T>
    //void get_arithmetic_space(T& x);
    void get_arithmetic(bool& x) {
        int y;
        get_arithmetic(y);
        CHECK(y == 0 || y == 1);
        x = y;
    }
    template<class T>
    void put_arithmetic(const T& x);
    void put_arithmetic(bool x) {
        put_arithmetic((int)x);
    }
};
*/
template<class AR>
struct DownpourSerializer {
    explicit DownpourSerializer(Archive<AR>& ar) : ar(ar) {
    }
    template<class T>
    DownpourSerializer<AR>& operator, (const T& x) {
        ar << x;
        return *this;
    }
    Archive<AR>& ar;
};

template<class AR>
struct DownpourDeserializer {
    explicit DownpourDeserializer(Archive<AR>& ar) : ar(ar) {
    }
    template<class T>
    DownpourDeserializer<AR>& operator, (T& x) {
        ar >> x;
        return *this;
    }
    Archive<AR>& ar;
};

#define MIO_DEFINE_SIMPLE_SERIALIZER(CLASSNAME, FIELDS...) \
    template<class AR> \
    void _downpour_serialize_internal_(::downpour::Archive<AR>& _downpour_ar_) const { \
        ::downpour::DownpourSerializer<AR> _downpour_serializer_(_downpour_ar_); \
        _downpour_serializer_, FIELDS; \
    } \
    template<class AR> \
    void _downpour_deserialize_internal_(::downpour::Archive<AR>& _downpour_ar_) { \
        ::downpour::DownpourDeserializer<AR> _downpour_deserializer_(_downpour_ar_); \
        _downpour_deserializer_, FIELDS; \
    } \
    template<class AR> \
    friend ::downpour::Archive<AR>& operator<<(::downpour::Archive<AR>& ar, const CLASSNAME& x) { \
        x._downpour_serialize_internal_(ar); \
        return ar; \
    } \
    template<class AR> \
    friend ::downpour::Archive<AR>& operator>>(::downpour::Archive<AR>& ar, CLASSNAME& x) { \
        x._downpour_deserialize_internal_(ar); \
        return ar; \
    }

template<class AR, class T, size_t N>
Archive<AR>& operator<<(Archive<AR>& ar, const T(&p)[N]) {
    for (size_t i = 0; i < N; i++) {
        ar << p[i];
    }

    return ar;
}

template<class AR, class T, size_t N>
Archive<AR>& operator>>(Archive<AR>& ar, T(&p)[N]) {
    for (size_t i = 0; i < N; i++) {
        ar >> p[i];
    }

    return ar;
}

template<class AR, class T>
Archive<AR>& operator<<(Archive<AR>& ar, const std::vector<T>& p) {
    ar << (size_t)p.size();

    for (const auto & x : p) {
        ar << x;
    }

    return ar;
}

template<class AR, class T>
Archive<AR>& operator>>(Archive<AR>& ar, std::vector<T>& p) {
    p.resize(ar.template get<size_t>());

    for (auto & x : p) {
        ar >> x;
    }

    return ar;
}

template<class AR, class T>
Archive<AR>& operator<<(Archive<AR>& ar, const std::valarray<T>& p) {
    ar << (size_t)p.size();

    for (const auto & x : p) {
        ar << x;
    }

    return ar;
}

template<class AR, class T>
Archive<AR>& operator>>(Archive<AR>& ar, std::valarray<T>& p) {
    p.resize(ar.template get<size_t>());

    for (auto & x : p) {
        ar >> x;
    }

    return ar;
}

inline BinaryArchive& operator<<(BinaryArchive& ar, const std::string& s) {
    ar << (size_t)s.length();
    ar.write(&s[0], s.length());
    return ar;
}

inline BinaryArchive& operator>>(BinaryArchive& ar, std::string& s) {
    size_t len = ar.template get<size_t>();
    ar.prepare_read(len);
    s.assign(ar.cursor(), len);
    ar.advance_cursor(len);
    return ar;
}

template<class AR, class T1, class T2>
Archive<AR>& operator<<(Archive<AR>& ar, const std::pair<T1, T2>& x) {
    return ar << x.first << x.second;
}

template<class AR, class T1, class T2>
Archive<AR>& operator>>(Archive<AR>& ar, std::pair<T1, T2>& x) {
    return ar >> x.first >> x.second;
}

template<class AR, class... T>
Archive<AR>& serialize_tuple(Archive<AR>& ar, const std::tuple<T...>& x,
        std::integral_constant<size_t, 0> n) {
    return ar;
}

template<class AR, class... T, size_t N>
Archive<AR>& serialize_tuple(Archive<AR>& ar, const std::tuple<T...>& x,
        std::integral_constant<size_t, N> n) {
    return serialize_tuple(ar, x, std::integral_constant < size_t, N - 1 > ()) << std::get < N - 1 > (x);
}

template<class AR, class... T>
Archive<AR>& operator<<(Archive<AR>& ar, const std::tuple<T...>& x) {
    const size_t size = std::tuple_size<std::tuple<T...>>::value;
    return serialize_tuple(ar, x, std::integral_constant<size_t, size>());
}

template<class AR, class... T>
Archive<AR>& deserialize_tuple(Archive<AR>& ar, std::tuple<T...>& x,
        std::integral_constant<size_t, 0> n) {
    return ar;
}

template<class AR, class... T, size_t N>
Archive<AR>& deserialize_tuple(Archive<AR>& ar, std::tuple<T...>& x,
        std::integral_constant<size_t, N> n) {
    return deserialize_tuple(ar, x, std::integral_constant < size_t,
            N - 1 > ()) >> std::get < N - 1 > (x);
}

template<class AR, class... T>
Archive<AR>& operator>>(Archive<AR>& ar, std::tuple<T...>& x) {
    const size_t size = std::tuple_size<std::tuple<T...>>::value;
    return deserialize_tuple(ar, x, std::integral_constant<size_t, size>());
}

#define PSLIB_REPEAT(MAP_TYPE, RESERVE_STATEMENT) \
    template<class AR, class KEY, class VALUE, class... ARGS> \
    Archive<AR>& operator<<(Archive<AR>& ar, const MAP_TYPE<KEY, VALUE, ARGS...>& p) { \
        ar << (size_t)p.size(); \
        for (auto it = p.begin(); it != p.end(); ++it) { \
            ar << *it; \
        } \
        return ar; \
    } \
    template<class AR, class KEY, class VALUE, class... ARGS> \
    Archive<AR>& operator>>(Archive<AR>& ar, MAP_TYPE<KEY, VALUE, ARGS...>& p) { \
        size_t size = ar.template get<size_t>(); \
        p.clear(); \
        RESERVE_STATEMENT; \
        for (size_t i = 0; i < size; i++) { \
            p.insert(ar.template get<std::pair<KEY, VALUE>>()); \
        } \
        return ar; \
    }

PSLIB_REPEAT(std::map,)
PSLIB_REPEAT(std::multimap,)
PSLIB_REPEAT(std::unordered_map, p.reserve(size))
PSLIB_REPEAT(std::unordered_multimap, p.reserve(size))

#undef PSLIB_REPEAT

#define PSLIB_REPEAT(SET_TYPE, RESERVE_STATEMENT) \
    template<class AR, class KEY, class... ARGS> \
    Archive<AR>& operator<<(Archive<AR>& ar, const SET_TYPE<KEY, ARGS...>& p) { \
        ar << (size_t)p.size(); \
        for (auto it = p.begin(); it != p.end(); ++it) { \
            ar << *it; \
        } \
        return ar; \
    } \
    template<class AR, class KEY, class... ARGS> \
    Archive<AR>& operator>>(Archive<AR>& ar, SET_TYPE<KEY, ARGS...>& p) { \
        size_t size = ar.template get<size_t>(); \
        p.clear(); \
        RESERVE_STATEMENT; \
        for (size_t i = 0; i < size; i++) { \
            p.insert(ar.template get<KEY>()); \
        } \
        return ar; \
    }


PSLIB_REPEAT(std::set,)
PSLIB_REPEAT(std::multiset,)
PSLIB_REPEAT(std::unordered_set, p.reserve(size))
PSLIB_REPEAT(std::unordered_multiset, p.reserve(size))

#undef PSLIB_REPEAT

}
}
#endif //
