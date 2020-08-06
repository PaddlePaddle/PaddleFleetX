#include "common.h"
#include "encode.h"

#include <assert.h>

namespace paddle {
namespace communication {
namespace dgc{

#define FABS(a) ((a!=-INFINITY) ? fabs(a) : a)

const int CUDA_NUM_THREADS = 512;

static unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor) {
  return ((dividend % divisor) == 0) ?
         (dividend / divisor) :
         (dividend / divisor + 1);
}

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

static void getNumBlocksAndThreads(int n, int &blocks, int &threads) {
  if (n == 1) {
    threads = 1;
    blocks = 1;
  } else {
    threads = (n < MAX_THREADS) ? nextPow2(n / 2) : MAX_THREADS;
    blocks = max(1, n / (threads * 2));
  }
  blocks = min(MAX_BLOCKS, blocks);

  assert(blocks * threads < n);
}

template <typename T>
__global__ void KeGetThreadCountByThreshold(const T* idata, int* odata, int count, T* threshold) 
{
  const int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= count) return;

  int cnt = 0;
  T kth = *threshold;
  for (int i = id; i < count; i += gridDim.x * blockDim.x) {
    if (FABS(idata[i]) >= kth) {
      cnt++;
    }
  }

  odata[id] = cnt;
}

template <typename T>
__global__ void KeGetRowsCount(const T* idata, int* odata, int count) 
{
  const int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= count) return;

  int cnt = 0;
  for (int i = id; i < count; i += gridDim.x * blockDim.x) {
    if (idata[i] > 0) {
      cnt++;
    }
  }

  odata[id] = cnt;
}

__global__ void KePrefixSum(int *data, int width, int *partial_sums=NULL) {
  extern __shared__ int shm[];
  int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
  int lane_id = id % warpSize;
  int warp_id = threadIdx.x / warpSize;
  int value = data[id];

#pragma unroll
  for (int i = 1; i <= width; i*=2) {
    unsigned int mask = 0xffffffff;
    int n = CudaShuffleUpSync(mask, value, i, width);
    if (lane_id >= i) value += n;
  }

  if (threadIdx.x % warpSize == warpSize-1) {
    shm[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
    int warp_sum = shm[lane_id];
    int mask = (1<<(blockDim.x / warpSize)) - 1;
    for (int i  = 1; i <= (blockDim.x / warpSize); i *= 2) {
      int n = CudaShuffleUpSync(mask, warp_sum, i, (blockDim.x / warpSize));
      if (lane_id >= i) warp_sum += n;
    }
    shm[lane_id] = warp_sum;
  }
  __syncthreads();

  int blockSum = 0;
  if (warp_id > 0) {
    blockSum = shm[warp_id-1];
  }
  value += blockSum;
  data[id] = value;

  if (partial_sums != NULL && threadIdx.x == blockDim.x-1) {
    partial_sums[blockIdx.x] = value;
  }
}

__global__ void KeGlobalPrefixSum(int *data, int *partial_sums, int len) {
  __shared__ int buf;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= len) return;
  if (threadIdx.x == 0) {
    buf = partial_sums[blockIdx.x];
  }
  __syncthreads();
  data[id] += buf;
}

template <typename T>
__global__ void KeEncodeInit(T* value, int* index, int k) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if(id >= k){
      return;
  }

  for (int i = id; i < k; i += gridDim.x * blockDim.x) {
    index[i] = -1;
    value[i] = -1;
  }
}

template <typename T>
__global__ void KeEncode(const T *data, int count, int *scan, T* value, int* index, T* threshold, int k) {
  int id = blockDim.x*blockIdx.x+threadIdx.x;
  int start = 0;
  if (id == 0) {
    start = 0;
  } else {
    start = scan[id-1];
  }
  __syncthreads();

  T kth = *threshold;
  int offset = start;

  for (int i = id; i < count; i += gridDim.x*blockDim.x) {
    T val = data[i];
    if (offset < k && FABS(val) > kth) {
      index[offset] = i;
      value[offset] = val;
      offset ++;
    }
  }
}

template <typename T>
__global__ void KeGetIndex(const T *data, int count, int *scan, int64_t* index, int64_t* nnz) {
  int id = blockDim.x*blockIdx.x + threadIdx.x;
  int offset = (id == 0) ? 0 : scan[id-1];
  if (id == 0) *nnz = scan[gridDim.x*blockDim.x - 1];

  for (int i = id; i < count; i += gridDim.x*blockDim.x) {
    T val = data[i];
    if (val > 0) {
      index[offset] = i;
      offset++;
    }
  }
}

template <typename T>
__global__ void KeMask(int* index, int k, T* data, int count) {
  int id = blockDim.x*blockIdx.x+threadIdx.x;
  for (int i = id; i < k; i += gridDim.x*blockDim.x) {
    int idx = index[i];
    if (idx >= count || idx < 0) {
      continue;
    }
    data[idx] = 0;
  }
}

void dense2idx(int64_t* index, int64_t* nnz, unsigned char* input,
               int* thr_cnt, int count, cudaStream_t stream) {
  int blocks, threads;
  getNumBlocksAndThreads(count, blocks, threads);
  int smemSize = sizeof(float) * threads;
  int p_threads = min(blocks, threads);
  int p_blocks = iDivUp(blocks, p_threads);

  KeGetRowsCount<unsigned char><<<blocks, threads, smemSize, stream>>>(input, thr_cnt, count);

  int* part = thr_cnt + threads * blocks;
  KePrefixSum<<<blocks, threads, smemSize, stream>>>(thr_cnt, 32, part);
  KePrefixSum<<<p_blocks, p_threads, smemSize, stream>>>(part, 32);
  KeGlobalPrefixSum<<<blocks-1, threads, 0, stream>>>(thr_cnt + threads, part, count);

  CUDA_CHECK(cudaMemsetAsync(static_cast<void*>(index), -1, count*sizeof(int64_t), stream));
  KeGetIndex<unsigned char><<<blocks, threads, 0, stream>>>(input, count, thr_cnt, index, nnz);
}

void dense2coo(void* encode, float * input, float* threshold, int* thr_cnt, int count, int k, cudaStream_t stream) {
  int blocks, threads;
  getNumBlocksAndThreads(count, blocks, threads);
  int smemSize = sizeof(float) * threads;
  int p_threads = min(blocks, threads);
  int p_blocks = iDivUp(blocks, p_threads);

  KeGetThreadCountByThreshold<float><<<blocks, threads, smemSize, stream>>>(input, thr_cnt, count, threshold);
  int* part = thr_cnt + threads * blocks;
  KePrefixSum<<<blocks, threads, smemSize, stream>>>(thr_cnt, 32, part);
  KePrefixSum<<<p_blocks, p_threads, smemSize, stream>>>(part, 32);
  KeGlobalPrefixSum<<<blocks-1, threads, 0, stream>>>(thr_cnt + threads, part, count);
  int* index = static_cast<int*>(encode);
  float* value = static_cast<float*>(encode) + k;
  KeEncodeInit<float><<<blocks, threads, 0, stream>>>(value, index, k);
  KeEncode<float><<<blocks, threads, 0, stream>>>(input, count, thr_cnt, value, index, threshold, k);
}

void mask(void* encode, int count, int k, float* input, cudaStream_t stream, float* moment) {
  int* index = static_cast<int*>(encode);
  KeMask<float><<<GET_BLOCKS(k), CUDA_NUM_THREADS, 0, stream>>>(
                               index, k, input, count);
  if (moment != NULL) {
    KeMask<float><<<GET_BLOCKS(k), CUDA_NUM_THREADS, 0, stream>>>(
                               index, k, moment, count);
  }
}

}  // namespace dgc
}  // namespace communication
}  // namespace paddle
