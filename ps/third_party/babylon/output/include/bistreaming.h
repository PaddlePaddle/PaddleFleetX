#ifndef BAIDU_FEED_MLARCH_BABYLON_BISTREAMING_H
#define BAIDU_FEED_MLARCH_BABYLON_BISTREAMING_H

#include <memory>
#include <iostream>
#include <unordered_map>
#include <comlog/comlog.h>
#include <baidu/feed/mlarch/babylon/expect.h>

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

// return 0  : 正常读到一个kv
// return 1  : 读到文件结尾
// return <0 : 读取异常
inline int32_t read_key_value_from_bistreaming(::std::string& key, ::std::string& value,
    ::std::istream& is = ::std::cin) noexcept {
    uint32_t length = 0;
    if (unlikely(!is.read(reinterpret_cast<char*>(&length), 4))) {
        return !is.bad() && is.gcount() == 0 ? 1 : -1;
    }

    key.resize(length);
    if (likely(length > 0)) {
        if (unlikely(!is.read(&key[0], length))) {
            CFATAL_LOG(" read key error");
            return -1;
        }
    }

    if (unlikely(!is.read(reinterpret_cast<char*>(&length), 4))) {
        CFATAL_LOG(" read value length error");
        return -1;
    }

    value.resize(length);
    if (likely(length > 0)) {
        if (unlikely(!is.read(&value[0], length))) {
            CFATAL_LOG(" read value error");
            return -1;
        }
    }
    return 0;
}

// return 0  : 正常写出一个kv
// return <0 : 写出异常
inline int32_t emit_key_value_to_bistreaming(const ::std::string& key, const ::std::string& value,
    ::std::ostream& os = ::std::cout) noexcept {
    uint32_t length = key.length();
    if (unlikely(!os.write(reinterpret_cast<char*>(&length), 4))) {
        CFATAL_LOG(" write key length error");
        return -1;
    }

    if (unlikely(!os.write(key.data(), length))) {
        CFATAL_LOG(" write key error");
        return -1;
    }

    length = value.length();
    if (unlikely(!os.write(reinterpret_cast<char*>(&length), 4))) {
        CFATAL_LOG(" write value length error");
        return -1;
    }

    if (unlikely(!os.write(value.data(), length))) {
        CFATAL_LOG(" write value error");
        return -1;
    }
    return 0;
}

// User-Defined Streaming Counters
// The line must have the following format: reporter:counter:group,counter,amount
// group: string, group name
// counter: string, counter name
// amount: int, counter value
inline void incr_counter(const std::string& group, const std::string& counter, const uint64_t& amount) {
    std::cerr << "reporter:counter:" << group << "," << counter << "," << amount << std::endl;
}


// Streaming Counter Wrapper
class StreamingCounter {
public:
    StreamingCounter() {}
    virtual ~StreamingCounter() {}

    void increment(const std::string& group, const std::string& counter, const uint64_t& amount) {
        auto group_iter = _collector.find(group);
        if (group_iter != _collector.end()) {
            auto& counter_map = group_iter->second;
            counter_map[counter] += amount;
        } else {
            std::unordered_map<std::string, uint64_t> counter_map;
            counter_map.emplace(counter, amount);
            _collector.emplace(group, std::move(counter_map));
        }
    }

    void increment_one(const std::string& group, const std::string& counter) {
        increment(group, counter, 1);
    }

    void submit() {
        for (auto const& group : _collector) {
            for (auto const& counter : group.second) {
                incr_counter(group.first, counter.first, counter.second);
            }
        }
    }

    void reset() {
        _collector.clear();
    }

private:
    std::unordered_map<std::string, std::unordered_map<std::string, uint64_t>> _collector;
};

}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif  // BAIDU_FEED_MLARCH_BABYLON_BISTREAMING_H
