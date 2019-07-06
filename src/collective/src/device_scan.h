#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common.h"

namespace paddle {
namespace communication {
namespace dgc{

#define WARP_SIZE 32

///************************* scan **********************///
///  scan,   inclusive scan
///  exscan, exclusive scan
template<typename T>
__device__ inline T warp_scan(T value, int lane_id) {
#pragma unroll
  for (int i = 1; i < WARP_SIZE; i *= 2) {
    unsigned int mask = 0xffffffff;
    T n = CudaShuffleUpSync(mask, value, i, WARP_SIZE);
    if (lane_id >= i) value += n;
  }
  return value;
}

template<typename T, int BlockSize>
__device__ inline T block_warp_scan(T value, int lane_id) {
  constexpr int NumWarp = BlockSize / WARP_SIZE;
#pragma unroll
  for (int i = 1; i < NumWarp; i *= 2) {
    unsigned int mask = (1<<NumWarp) - 1;
    T n = CudaShuffleUpSync(mask, value, i, NumWarp);
    if (lane_id >= i) value += n;
  }
  return value;
}

template<typename T, int BlockSize>
__device__ inline
T inclusive_scan(T idata) {
  constexpr int NumWarp = BlockSize / WARP_SIZE;
  __shared__ volatile T sdata[NumWarp];

  int lane_id = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;

  // step 1, get warp inclusive scan value
  T oval = warp_scan(idata, lane_id);

  // save warp scan into share memory
  if (lane_id == WARP_SIZE - 1) sdata[warp_id] = oval;
  __syncthreads();

  // step 2, get block inclusive scan value
  if (warp_id == 0 && lane_id < NumWarp) {
    sdata[lane_id] = block_warp_scan<T, BlockSize>(sdata[lane_id], lane_id);
  }
  __syncthreads();

  // exclusive scan value of current warp
  T block_sum = 0;
  if (warp_id > 0) block_sum = sdata[warp_id - 1];

  // inclusive scan value of current thread
  oval += block_sum;
  return oval;
}

template<typename T4, typename T, int BlockSize>
__device__ inline
T4 scan4(T4 idata4, T pre_sum=0) {
  // step 1, get thread inclusive scan value
  idata4.y += idata4.x;
  idata4.z += idata4.y;
  idata4.w += idata4.z;

  // step 2, get inclusive scan value of current thread
  T oval = inclusive_scan<T, BlockSize>(idata4.w);
  // exclusive scan value of current thread
  oval = oval - idata4.w + pre_sum;

  idata4.x += oval;  idata4.y += oval;
  idata4.z += oval;  idata4.w += oval;
  return idata4;
}

}  // namespace dgc
}  // namespace communication
}  // namespace paddle
