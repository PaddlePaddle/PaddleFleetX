#pragma once

#include <cuda_runtime.h>
#include <cstdio>

#include "nccl.h"

namespace paddle {
namespace communication {
namespace dgc{

#define LOGERR(format, args...)                   \
    (fprintf(stderr, "[%s:%d:%s] " format "\n",   \
    __FILE__, __LINE__, __FUNCTION__, ##args))

#define CUDA_CHECK(cmd) do {                  \
    cudaError_t e = cmd;                      \
    if (e != cudaSuccess) {                   \
        LOGERR("Cuda failure - %s",           \
               cudaGetErrorString(e));        \
        exit(EXIT_FAILURE);                   \
    }                                         \
} while (0)

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

#if CUDART_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

template <typename T>
__forceinline__ __device__ T CudaShuffleSync(unsigned mask, T val, int src_line,
                                             int width = 32) {
#if CUDART_VERSION < 9000
  return __shfl(val, src_line, width);
#else
  return __shfl_sync(mask, val, src_line, width);
#endif
}

template <typename T>
__forceinline__ __device__ T CudaShuffleUpSync(unsigned mask, T val, int delta, 
                                               int width = 32) {
#if CUDART_VERSION < 9000
  return __shfl_up(val, delta, width);
#else
  return __shfl_up_sync(mask, val, delta, width);
#endif
}

#define MAX_THREADS 256
#define MAX_BLOCKS  128

}  // namespace dgc
}  // namespace communication
}  // namespace paddle
