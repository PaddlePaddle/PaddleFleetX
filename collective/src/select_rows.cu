#include "common.h"
#include "ncclwrap.h"
#include "dgc.h"
#include "encode.h"

#include <stdio.h>
#include <assert.h>
#include <pthread.h>

namespace paddle {
namespace communication {
namespace dgc {

#define KER_CHECK(cmd) do {  \
  if (!(cmd)) { return; }    \
} while(0)

#define BLOCK_SIZE 256
#define MAX_BLOCK 128
#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

typedef struct {
  int nnz;
  int* rows_idx;
  float* dst;
}rowsRet;

__global__
void markRowsKernel(int8_t* rows_bitmap, int* rows_posmap, const size_t count,
                    const int* rows_idx, const size_t nnz) {
  const int tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  const int N = BLOCK_SIZE * MAX_BLOCK;

  for (int i = tid; i < nnz; i += N) {
    int idx = rows_idx[i];
    KER_CHECK(idx >= 0 && idx < count);
    rows_bitmap[idx] = 1;  // mark which idx has value
    rows_posmap[idx] = i;  // record the position of idx in src
  }
}

//template <typename T>
//__global__ void getRowsCountKernel(int* output, const T* input, const int count) {
//  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
//  if (tid >= count) return;
//
//  int cnt = 0;
//  for (int i = tid; i < count; i += gridDim.x * blockDim.x) {
//    if (input[i] > 0) ++cnt;
//  }
//
//  // get block inclusive scan
//  int sum = inclusive_scan<BLOCK_SIZE, int>(cnt);
//  output[tid] = sum;
//}

template<typename T>
__global__ void gatherRowsValKernel(T* dst, const T* src, const int row_width, const int* rows_idx, const int nnz, int* rows_posmap) {
  const int tid = threadIdx.x;
  int bid = blockIdx.x;

  const size_t inc = MAX_BLOCK * row_width;
  T* dst0 = dst + bid * row_width;

  // one block copy one row
  while (bid < nnz) {
    int idx = rows_idx[bid];
    int pos = rows_posmap[idx];
    
    if (pos == -1) {
      for (size_t i = tid; i < row_width; i += BLOCK_SIZE) {
        dst0[i] = 0; // set to 0
      }
    } else {
//      if (tid == 0) printf("debug bid=%d idx=%d pos=%d\n", bid, idx, pos);
      const T* src0 = src + pos * row_width; 
      for (size_t i = tid; i < row_width; i += BLOCK_SIZE) {
        dst0[i] = src0[i]; // copy from src
      }
    }

    bid += MAX_BLOCK;
    dst0 += inc; 
  }
}

static void busy_wait(volatile int* flag) {
  while (*flag == -1) sched_yield();
}

template<typename T>
static void gMalloc(T** ptr, size_t nelem, GpuAllocator* ator) {
  void* p = ator->allocator(sizeof(T) * nelem);
  assert(p != NULL);
  *ptr = static_cast<T*>(p);
  return;
}

rowsRet selectRowsAllReduce(const std::vector<int>& rows_idx, const int count,
    const float* src, const int row_width,
    GpuAllocator* ator, ncclComm_t comm, cudaStream_t stream) {
  const int nnz = rows_idx.size();

  int* d_rows_idx = NULL;
  int8_t* d_rows_bitmap = NULL;
  int* d_rows_posmap = NULL; // used to record the position of idx in src
  gMalloc(&d_rows_idx, count, ator);
  gMalloc(&d_rows_bitmap, count, ator);
  gMalloc(&d_rows_posmap, count, ator); 

  CUDA_CHECK(cudaMemcpyAsync(d_rows_idx, rows_idx.data(),
          sizeof(int)*nnz, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(static_cast<void*>(d_rows_bitmap), 0, count*sizeof(int8_t), stream));
  CUDA_CHECK(cudaMemsetAsync(static_cast<void*>(d_rows_posmap), -1, count*sizeof(int), stream));

  int blocks = min(DIVUP(nnz, BLOCK_SIZE), MAX_BLOCK);
  markRowsKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d_rows_bitmap, d_rows_posmap, count, d_rows_idx, nnz);

  //// merge rows
  warpNcclAllReduce(d_rows_bitmap, d_rows_bitmap, count, ncclInt8, ncclMax, comm, stream);

  //// copy src to dst, then allReduce
  int c_nnz = -1;
  int* d_thr_cnt = static_cast<int*>(ator->allocator(get_buffer_size(count)));
  // get d_rows_idx and c_nnz from merged d_rows_bitmap
  dense2idx(d_rows_idx, &c_nnz, d_rows_bitmap, d_thr_cnt, count, stream);

  busy_wait(&c_nnz);

  float* dst = NULL;
  gMalloc(&dst, c_nnz * row_width, ator);

  // copy src to dst
  blocks = min(c_nnz, MAX_BLOCK);
  gatherRowsValKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, row_width,
                                            d_rows_idx, c_nnz, d_rows_posmap);

  warpNcclAllReduce(dst, dst, c_nnz * row_width, ncclFloat, ncclSum, comm, stream);
  rowsRet ret = {c_nnz, d_rows_idx, dst};
  return ret;
}

static void args_check(const std::vector<int>& rows_idx, const int count,
    const float* src, const int row_width, std::vector<int>* dst_rows, float** dst,
    GpuAllocator* ator) {
  assert(rows_idx.size() > 0 && rows_idx.size() < count);
  assert(src != NULL);
  assert(row_width > 0);
  assert(dst_rows != NULL);
  assert(dst != NULL);
  assert(ator != NULL);
}


int allReduceSelectedRows(const int max_rows_height, const std::vector<int>& src_rows,
    const int row_width, const float* src_tensor, std::vector<int>* dst_rows,
    float** dst_tensor, GpuAllocator* allocator, ncclComm_t comm, cudaStream_t stream) {
  args_check(src_rows, max_rows_height, src_tensor, row_width, dst_rows, dst_tensor, allocator);

  int device = -1;
  warpNcclCommCuDevice(comm, &device);
  assert(device != -1);
  CudaDeviceGuard guard(device);

  rowsRet ret = selectRowsAllReduce(src_rows, max_rows_height,
                                    src_tensor, row_width, allocator, comm, stream);
  const int nnz = ret.nnz;
  dst_rows->resize(nnz);
  CUDA_CHECK(cudaMemcpyAsync(dst_rows->data(), ret.rows_idx,
                  sizeof(int)*nnz, cudaMemcpyDeviceToHost, stream));
  *dst_tensor = ret.dst;
  return true;
}

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

