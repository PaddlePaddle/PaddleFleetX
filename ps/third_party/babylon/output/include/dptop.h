#ifndef BAIDU_FEED_MLARCH_BABYLON_DPTOP_H
#define BAIDU_FEED_MLARCH_BABYLON_DPTOP_H

#include <Eigen/Core>
#include <baidu/feed/mlarch/babylon/expect.h>

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

// 一个点积结果，包含序号和点积值
struct DotProductResult {
    DotProductResult() noexcept : index(0), score(0.0) {}
    DotProductResult(size_t index, float score) noexcept : index(index), score(score) {}

    bool operator<(const DotProductResult& other) const noexcept {
        return score < other.score;
    }

    bool operator>(const DotProductResult& other) const noexcept {
        return score > other.score;
    }

    size_t index;
    float score;
};

// 点积结果堆
class DotProductResultHeap {
public:
    DotProductResultHeap() : _is_heap(false), _retain(0),
        _results(&_results_data), _results_data() {}

    // 实际存储引用传入的result
    void wrap(::std::vector<DotProductResult>& results) {
        _results = &results;
    }

    // 重置堆，清空内容，并重新设置最大保留数目
    void reset(size_t retain) {
        _is_heap = false;
        _retain = retain;
        _results->clear();
        _results->reserve(_retain + 1);
    }

    // 增加一个点积结果，保留值最大的retain个结果
    void add(size_t index, float score) {
        if (unlikely(_results->size() < _retain)) {
            _results->emplace_back(index, score);
        } else {
            if (!_is_heap) {
                ::std::make_heap(_results->begin(), _results->end(),
                    ::std::greater<DotProductResult>());
                _is_heap = true;
            }

            const auto& top =(*_results)[0];
            if (score > top.score) {
                _results->emplace_back(index, score);
                ::std::pop_heap(_results->begin(), _results->end(),
                    ::std::greater<DotProductResult>());
                _results->pop_back();
            }
        }
    }

    // 合并两个点积结果堆，保留并集中值最大的retain个结果
    void merge(const DotProductResultHeap& other, size_t offset) {
        for (const auto& result : *(other._results)) {
            add(result.index + offset, result.score);
        }
    }

    // 最终结果从堆变成排序数组
    void sort() {
        if (_is_heap) {
            ::std::sort_heap(_results->begin(), _results->end(),
                ::std::greater<DotProductResult>());
            _is_heap = false;
        } else {
            ::std::sort(_results->begin(), _results->end(),
                ::std::greater<DotProductResult>());
        }
    }

    // 返回内部的结果数组
    ::std::vector<DotProductResult>& results() {
        return *_results;
    }

private:
    bool _is_heap;
    size_t _retain;
    ::std::vector<DotProductResult>* _results;
    ::std::vector<DotProductResult> _results_data;
};

// 进行D维度的点积计算
// return = v0[0 - D] * v1[0 - D]
template<size_t D>
inline __attribute__((always_inline)) float dot_product(const float* v0, const float* v1) {
    typedef ::Eigen::Map<const ::Eigen::Matrix<float, D, 1>> MVector;
    float result = MVector(v0).dot(MVector(v1));
    return result;
}

// 进行M个D维度的点积计算，M == 1的特例
// result[0] = v0[0 - D] * v1[0 - D]
template<size_t D, size_t M>
inline __attribute__((always_inline)) typename std::enable_if<M == 1, void>::type
mdot_product(float* result, const float* v0, const float* v1) {
    *result = dot_product<D>(v0, v1);
}

// 进行M个D维度的点积计算，M > 1的情况
// 最终递归展开到M == 1的特例
// result[0] = v0[0 - D] * v1[0 - D]
// result[1] = v0[D - 2*D] * v1[0 - D]
// ...
// result[M-1] = v0[(M-1)*D - M*D] * v1[0 - D]
template<size_t D, size_t M>
inline __attribute__((always_inline)) typename std::enable_if<(M > 1), void>::type
mdot_product(float* result, const float* v0, const float* v1) {
    *result = dot_product<D>(v0, v1);
    mdot_product<D, M - 1>(result + 1, v0 + D, v1);
}

// 同时进行M路top运算，得到M个点积结果堆，堆中结果维持堆状态
// 如果要得到排序结果需要heap.sort()
// heaps[0] = top(v0[0 - D] * v1[0 - size*D], limit)
// heaps[1] = top(v0[D - 2*D] * v1[0 - size*D], limit)
// ...
// heaps[M] = top(v0[(M-1)*D - M*D] * v1[0 - size*D], limit)
template<uint64_t D, uint64_t M>
void top_dot_product(::std::vector<DotProductResultHeap>& heaps,
    size_t limit, const float* v0, const float* v1, size_t size) {
    if (unlikely(heaps.size() < M)) {
        heaps.resize(M);
    }
    for (size_t i = 0; i < M; ++i) {
        heaps[i].reset(limit);
    }

    float score[M];
    const float* end = v1 + size * D;
    for (size_t index = 0; v1 < end; v1 += D, index += 1) {
        mdot_product<D, M>(score, v0, v1);
        for (size_t i = 0; i < M; ++i) {
            heaps[i].add(index, score[i]);
        }
    }
}

// 同时进行M路top运算，得到M个已经排序的点积结果
template<uint64_t D, uint64_t M>
void top_dot_product(::std::vector<::std::vector<DotProductResult>>& results,
    size_t limit, const float* v0, const float* v1, size_t size) {
    static thread_local ::std::vector<DotProductResultHeap> heaps;
    if (unlikely(heaps.size() < M)) {
        heaps.resize(M);
    }
    if (unlikely(results.size() < M)) {
        results.resize(M);
    }
    for (size_t i = 0; i < M; ++i) {
        heaps[i].wrap(results[i]);
    }

    top_dot_product<D, M>(heaps, limit, v0, v1, size);

    for (size_t i = 0; i < M; ++i) {
        heaps[i].sort();
    }
}

#ifdef BABYLON_AVX512
extern template void top_dot_product<32, 1>(::std::vector<::std::vector<DotProductResult>>& results,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<32, 2>(::std::vector<::std::vector<DotProductResult>>& results,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<32, 3>(::std::vector<::std::vector<DotProductResult>>& results,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<32, 4>(::std::vector<::std::vector<DotProductResult>>& results,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<32, 1>(::std::vector<DotProductResultHeap>& heaps,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<32, 2>(::std::vector<DotProductResultHeap>& heaps,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<32, 3>(::std::vector<DotProductResultHeap>& heaps,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<32, 4>(::std::vector<DotProductResultHeap>& heaps,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<64, 1>(::std::vector<::std::vector<DotProductResult>>& results,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<64, 2>(::std::vector<::std::vector<DotProductResult>>& results,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<64, 3>(::std::vector<::std::vector<DotProductResult>>& results,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<64, 4>(::std::vector<::std::vector<DotProductResult>>& results,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<64, 1>(::std::vector<DotProductResultHeap>& heaps,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<64, 2>(::std::vector<DotProductResultHeap>& heaps,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<64, 3>(::std::vector<DotProductResultHeap>& heaps,
    size_t limit, const float* v0, const float* v1, size_t size);
extern template void top_dot_product<64, 4>(::std::vector<DotProductResultHeap>& heaps,
    size_t limit, const float* v0, const float* v1, size_t size);
#endif

}  // babylon
}  // mlarch
}  // feed
}  // baidu

#endif  // BAIDU_FEED_MLARCH_BABYLON_DPTOP_H
