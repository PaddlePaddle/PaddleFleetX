#include "device_scan.h"
#include "ncclwrap.h"
#include "threshold.h"
#include <assert.h>

namespace paddle {
namespace communication {
namespace dgc {

#define BLOCK_SIZE 256
#define MAX_BLOCK 128

#define FABS(a) ((a!=-INFINITY) ? fabs(a) : a)

template<typename T, int BlockSize>
__global__ void calL1Norm(T* l1norm, const size_t chunk_width, const T* input, const size_t count,
                          bool* ipt_bitmap, const int nranks) {
  int chunks = DIVUP(count, chunk_width);

  const int tid = threadIdx.x;
  int bid = blockIdx.x;
 
  const size_t inc = MAX_BLOCK * chunk_width;
  size_t offset = bid * chunk_width;

  // one block process one chunk every iter
  while (bid < chunks) {
    const T* src = input + offset;
    size_t loop_count = min(chunk_width, count - offset);

    T sum = 0;
    for (int i = tid; i < loop_count; i += BlockSize) {
      sum += FABS(src[i]);
    }
    sum = inclusive_scan<T, BlockSize>(sum);
    if (threadIdx.x == BlockSize - 1) {
      sum /= loop_count;  // remain loop count may less than chunk_width
      l1norm[bid] = ipt_bitmap[bid] ? sum/nranks : sum;
    }
    bid += MAX_BLOCK;
    offset += inc;
  }
}

template<typename T, int BlockSize>
__global__ void gatherIptValKernel(T* dst, const size_t chunk_width, const T* src, const size_t count,
                                   const int* ipt_idx, const int k) {
  const int tid = threadIdx.x;
  int bid = blockIdx.x;

  const size_t inc = MAX_BLOCK * chunk_width;
  T* dst0 = dst + bid * chunk_width;

  // one block copy one chunk 
  while (bid < k) {
    int idx = ipt_idx[bid];
    const T* src0 = src + idx * chunk_width;
    size_t loop_count = min(chunk_width, count - idx * chunk_width);
    // Fixme. remain filled 0?
    for (size_t i = tid; i < loop_count; i += BlockSize) {
      dst0[i] = src0[i];
    }

    bid += MAX_BLOCK;
    dst0 += inc;
  }
}

// Todo. gather & scatter use template?
template<typename T, int BlockSize>
__global__ void scatterIptValKernel(T* dst, const size_t count, const T* src, const size_t chunk_width,
                                   const int* ipt_idx, const int k) {
  const int tid = threadIdx.x;
  int bid = blockIdx.x;

  const size_t inc = MAX_BLOCK * chunk_width;
  const T* src0 = src + bid * chunk_width;

  // one block copy one chunk 
  while (bid < k) {
    int idx = ipt_idx[bid];
    T* dst0 = dst + idx * chunk_width;
    size_t loop_count = min(chunk_width, count - idx * chunk_width);
    for (size_t i = tid; i < loop_count; i += BlockSize) {
      dst0[i] = src0[i];
    }

    bid += MAX_BLOCK;
    src0 += inc;
  }
}

static void csc_args_check(float* gradient, const size_t count,
                  float* ipt_chunks, const size_t chunk_width,
                  const int* ipt_idx, const int k) {
  assert(k > 0 && chunk_width > 0 && chunk_width * k <= count);
  assert(gradient != NULL);
  assert(ipt_chunks != NULL);
  assert(ipt_idx != NULL);
}

bool cscAllReduce(float* gradient, const size_t count,
                  float* ipt_chunks, const size_t chunk_width,
                  const int* ipt_idx, const int k,
                  ncclComm_t comm, cudaStream_t stream) {
  csc_args_check(gradient, count, ipt_chunks, chunk_width, ipt_idx, k);
  int device = -1;
  warpNcclCommCuDevice(comm, &device);
  assert(device != -1);
  CudaDeviceGuard guard(device);

  int blocks = min(k, MAX_BLOCK);
  gatherIptValKernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(ipt_chunks,
                                            chunk_width, gradient, count, ipt_idx, k);
  warpNcclAllReduce(ipt_chunks, ipt_chunks, k * chunk_width, ncclFloat, ncclSum, comm, stream);
  scatterIptValKernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(gradient,
                                            count, ipt_chunks, chunk_width, ipt_idx, k);
  return true;
}

static void ipt_args_check(int* ipt_idx, const int k, bool* ipt_bitmap,
                  float* norms, const size_t chunk_width,
                  const float* input, const size_t count) {
  assert(k > 0 && chunk_width > 0 && chunk_width * k <= count);
  assert(ipt_idx != NULL);
  assert(ipt_bitmap != NULL);
  assert(norms != NULL);
  assert(input != NULL);
}

// ipt_idx, out
bool calImportant(int* ipt_idx, const int k, bool* ipt_bitmap,
                  float* norms, const size_t chunk_width,
                  const float* input, const size_t count,  
                  ncclComm_t comm, cudaStream_t stream) {
  ipt_args_check(ipt_idx, k, ipt_bitmap, norms, chunk_width, input, count);
  int device = -1;
  warpNcclCommCuDevice(comm, &device);
  assert(device != -1);
  CudaDeviceGuard guard(device);

  int nranks = 0;
  warpNcclCommCount(comm, &nranks);

  int chunks = DIVUP(count, chunk_width);
  int blocks = min(MAX_BLOCK, chunks);
  calL1Norm<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(norms,
                           chunk_width, input, count, ipt_bitmap, nranks);
  
  warpNcclAllReduce(norms, norms, chunks, ncclFloat, ncclSum, comm, stream);

  cudaMemsetAsync(ipt_idx, -1, k * sizeof(int), stream);
  cudaMemsetAsync(ipt_bitmap, 0, chunks * sizeof(bool), stream);
  get_ipt_idx(ipt_idx, k, ipt_bitmap, norms, chunks, stream); 
  return true;
}
  
}  // namespace dgc
}  // namespace communication
}  // namespace 
