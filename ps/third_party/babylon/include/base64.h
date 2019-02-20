#ifndef BAIDU_FEED_MLARCH_BABYLON_BASE64_H
#define BAIDU_FEED_MLARCH_BABYLON_BASE64_H

#include <math.h>
#include <string>
#include <baidu/feed/mlarch/babylon/expect.h>

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

extern const uint32_t BASE64_DECODE_TABLE_1[65536];
extern const uint32_t BASE64_DECODE_TABLE_2[65536];

extern const uint16_t BASE64_ENCODE_TABLE_1[65536];
extern const uint16_t BASE64_ENCODE_TABLE_2[65536];

inline int32_t base64_encode(::std::string& dest, size_t dest_begin,
    const ::std::string& src, size_t src_begin, size_t src_size) noexcept {
    if (unlikely(src_size == 0)) {
        if (dest.size() > dest_begin) {
            dest.resize(dest_begin);
        }
        return 0;
    }
    size_t buffer_size = (src_size << 1) + 2;
    dest.resize(dest_begin + buffer_size);
    auto psrc = &src[src_begin];
    auto psrc_end = psrc + src_size;
    auto psrc_aligned_end = psrc_end - 2;
    auto pdest = &dest[dest_begin];
    auto pdest_begin = pdest;

    for (; psrc < psrc_aligned_end; psrc += 3, pdest += 4) {
        *(uint16_t*)pdest = BASE64_ENCODE_TABLE_1[*(uint16_t*)psrc];
        *(uint16_t*)(pdest + 2) = BASE64_ENCODE_TABLE_2[*(uint16_t*)(psrc + 1)];
    }

    switch (psrc_end - psrc) {
    case 0: {
            dest.resize(pdest - pdest_begin);
            break;
        }
    case 1: {
            *(uint16_t*)pdest = BASE64_ENCODE_TABLE_1[*(uint16_t*)psrc];
            pdest[2] = '=';
            pdest[3] = '=';
            dest.resize(pdest - pdest_begin + 4);
            break;
        }
    case 2: {
            *(uint16_t*)pdest = BASE64_ENCODE_TABLE_1[*(uint16_t*)psrc];
            *(uint16_t*)(pdest + 2) = BASE64_ENCODE_TABLE_2[*(uint16_t*)(psrc + 1)];
            pdest[3] = '=';
            dest.resize(pdest - pdest_begin + 4);
            break;
        }
    }
    return 0;
}

inline int32_t base64_encode(::std::string& dest, const ::std::string& src) noexcept {
    return base64_encode(dest, 0, src, 0, src.size());
}

inline int32_t base64_decode(::std::string& dest, size_t dest_begin,
    const ::std::string& src, size_t src_begin, size_t src_size) noexcept {
    if (unlikely((src_size & 0x03) != 0)) {
        return -1;
    } else if (unlikely(src_size == 0)) {
        if (dest.size() > dest_begin) {
            dest.resize(dest_begin);
        }
        return 0;
    }
    size_t buffer_size = src_size;
    dest.resize(dest_begin + buffer_size);
    auto psrc = &src[src_begin];
    auto psrc_end = psrc + src_size;
    auto psrc_aligned_end = psrc_end - 4;
    auto pdest = &dest[dest_begin];
    auto pdest_begin = pdest;

    for (; psrc < psrc_aligned_end; psrc += 4, pdest += 3) {
        uint32_t result = BASE64_DECODE_TABLE_1[*(uint16_t*)psrc] |
            BASE64_DECODE_TABLE_2[*(uint16_t*)(psrc + 2)];
        if (unlikely(result == 0xFFFFFFFF)) {
            if (dest.size() > dest_begin) {
                dest.resize(dest_begin);
            }
            return -1;
        }
        *(uint32_t*)pdest = result;
    }

    if (psrc[3] == '=') {
        if (psrc[2] == '=') {
            uint32_t result = BASE64_DECODE_TABLE_1[*(uint16_t*)psrc];
            if (unlikely(result == 0xFFFFFFFF)) {
                if (dest.size() > dest_begin) {
                    dest.resize(dest_begin);
                }
                return -1;
            }
            *(uint32_t*)pdest = result;
            dest.resize(pdest - pdest_begin + 1);
        } else {
            uint32_t result = BASE64_DECODE_TABLE_1[*(uint16_t*)psrc] |
                BASE64_DECODE_TABLE_2[psrc[2] | 0x4100];
            if (unlikely(result == 0xFFFFFFFF)) {
                if (dest.size() > dest_begin) {
                    dest.resize(dest_begin);
                }
                return -1;
            }
            *(uint32_t*)pdest = result;
            dest.resize(pdest - pdest_begin + 2);
        }
    } else {
        uint32_t result = BASE64_DECODE_TABLE_1[*(uint16_t*)psrc] |
            BASE64_DECODE_TABLE_2[*(uint16_t*)(psrc + 2)];
        if (unlikely(result == 0xFFFFFFFF)) {
            if (dest.size() > dest_begin) {
                dest.resize(dest_begin);
            }
            return -1;
        }
        *(uint32_t*)pdest = result;
        dest.resize(pdest - pdest_begin + 3);
    }
    return 0;
}

inline int32_t base64_decode(::std::string& dest, const ::std::string& src) noexcept {
    return base64_decode(dest, 0, src, 0, src.size());
}

}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif  // BAIDU_FEED_MLARCH_BABYLON_BASE64_H
