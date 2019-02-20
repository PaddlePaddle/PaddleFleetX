#ifndef BAIDU_FEED_MLARCH_BABYLON_MAP_H
#define BAIDU_FEED_MLARCH_BABYLON_MAP_H

#include <vector>
#include <inttypes.h>
#include <gflags/gflags.h>
#include <baidu/feed/mlarch/babylon/expect.h>

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

template <typename T>
class Comparator {
public:
    inline int compare(const T& left, const T& right) const noexcept {
        return left - right;
    }

    inline bool operator() (const T& left, const T& right) const noexcept {
        return compare(left, right) < 0;
    }
};

template <typename K, typename V, typename C = Comparator<K>,
         typename A = ::std::allocator<::std::pair<K, V>>>
class ReadOnlyMap {
public:
    typedef ::std::pair<K, V> value_type;
    typedef ::std::vector<value_type, A> container_type;
    typedef typename container_type::const_iterator const_iterator;

    ReadOnlyMap(const C& comparator = C(), const A& allocator = A()) noexcept :
        _elements(allocator), _comparator(comparator) {}

    ReadOnlyMap(std::initializer_list<value_type> init,
        const C& comparator = C(), const A& allocator = A()) noexcept :
        _elements(init, allocator), _comparator(comparator) {
        ::std::sort(_elements.begin(), _elements.end(), [this](
                const value_type& left, const value_type& right) {
            return _comparator(left.first, right.first);
        });
    }

    template <typename IT>
    ReadOnlyMap(IT first, IT last, const C& comparator = C(), const A& allocator = A()) noexcept :
        _elements(first, last, allocator), _comparator(comparator) {
        ::std::sort(_elements.begin(), _elements.end(), [this](
                const value_type& left, const value_type& right) {
            return _comparator(left.first, right.first);
        });
    }

    template <typename IT>
    void assign(IT first, IT last) noexcept {
        _elements.assign(first, last);
        ::std::sort(_elements.begin(), _elements.end(), [this](
                const value_type& left, const value_type& right) {
            return _comparator(left.first, right.first);
        });
    }

    inline const_iterator find(const K& key) const noexcept {
        const_iterator it = _elements.begin();
        int64_t size = _elements.size();
        while (size > 0) {
            int64_t middle = size >> 1;
            auto middle_it = it + middle;
            int comp_result = _comparator.compare(middle_it->first, key);
            if (comp_result == 0) {
                return middle_it;
            } else if (comp_result < 0) {
                it = middle_it + 1;
                size = size - middle - 1;
            } else {
                size = middle;
            }
        }
        return _elements.end();
    }

    inline const_iterator begin() const noexcept {
        return _elements.begin();
    }

    inline const_iterator end() const noexcept {
        return _elements.end();
    }

    inline void clear() noexcept {
        _elements.clear();
    }

private:
    container_type _elements;
    C _comparator;
};

template <>
int Comparator<::std::string>::compare(
    const ::std::string& left, const ::std::string& right) const noexcept {
    return left.compare(right);
};

}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif //BAIDU_FEED_MLARCH_BABYLON_MAP_H
