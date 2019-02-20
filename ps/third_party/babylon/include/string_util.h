#ifndef BAIDU_FEED_MLARCH_BABYLON_STRING_H
#define BAIDU_FEED_MLARCH_BABYLON_STRING_H

#include <stdarg.h>
#include <string>
#include <vector>
#include "expect.h"

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

// 用于::std::string的printf
// 格式化写入内容到str[0:max_size]中，str不要求预先分配空间
// 但要求max_size预估足够大小，否则可能会有越界错
// 返回实际写入字节数
inline int32_t strprintf(::std::string& str, size_t max_size,
    const char* format, ...) noexcept {
    if (unlikely(max_size == 0)) {
        str.clear();
        return 0;
    }

    va_list args;
    va_start(args, format);
    str.resize(max_size);
    auto ret = vsprintf(&str[0], format, args);
    str.resize(ret);
    va_end(args);
    return ret;
}

// 用于::std::string的printf
// 格式化写入内容到str[start:start + max_size]中，str不要求预先分配空间
// 但要求max_size预估足够大小，否则可能会有越界错
// 返回实际写入字节数
inline int32_t strprintf(::std::string& str, size_t start, size_t max_size,
    const char* format, ...) noexcept {
    if (unlikely(max_size == 0)) {
        str.resize(start);
        return 0;
    }

    va_list args;
    va_start(args, format);
    str.resize(start + max_size);
    auto ret = vsprintf(&str[start], format, args);
    str.resize(start + ret);
    va_end(args);
    return ret;
}

// String split function
// It uses reentrant variant of strtok.
// replace strtok with strspn avoid copy & 20% faster
inline void split(::std::vector<::std::string>& tokens, const ::std::string& str,
                  const ::std::string& delim = " ") noexcept {
    const char* pstr = str.c_str();
    size_t delim_size;
    size_t num = 0;
    do {
        auto token_size = strcspn(pstr, delim.c_str());
        if (num < tokens.size()) {
            tokens[num].assign(pstr, token_size);
        } else {
            tokens.emplace_back(pstr, token_size);
        }
        ++num;
        pstr += token_size;
        delim_size = strspn(pstr, delim.c_str());
        pstr += delim_size;
    } while (delim_size != 0);

    tokens.resize(num);
}

inline void split(::std::vector<::std::string>& tokens, const ::std::string& str,
    const char delim = ' ') {
    size_t begin = 0;
    size_t num = 0;
    size_t end = 0;

    do {
        size_t length;
        end = str.find(delim, begin);
        if (likely(end != ::std::string::npos)) {
            length = end - begin;
        } else {
            length = str.size() - begin;
        }

        if (num < tokens.size()) {
            tokens[num].assign(&str[begin], length);
        } else {
            tokens.emplace_back(&str[begin], length);
        }
        ++num;

        begin = end + 1;
    } while (end != ::std::string::npos);

    tokens.resize(num);
}

}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif  // BAIDU_FEED_MLARCH_BABYLON_STRING_H
