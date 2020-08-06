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

__global__
void markRowsKernel(unsigned char* rows_bitmap, int64_t* rows_posmap, const size_t count,
                    const int64_t* rows_idx, const size_t nnz) {
  const int tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  const int N = BLOCK_SIZE * MAX_BLOCK;

  for (int i = tid; i < nnz; i += N) {
    int idx = rows_idx[i];
    KER_CHECK(idx >= 0 && idx < count);
    rows_bitmap[idx] = 1;  // mark which idx has value
    rows_posmap[idx] = i;  // record the position of idx in src
  }
}

template<typename T>
__global__ void gatherRowsValKernel(T* dst, const T* src, const int64_t row_width,
             const int64_t* rows_idx, const int64_t nnz, const int64_t* rows_posmap) {
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
      const T* src0 = src + pos * row_width; 
      for (size_t i = tid; i < row_width; i += BLOCK_SIZE) {
        dst0[i] = src0[i]; // copy from src
      }
    }

    bid += MAX_BLOCK;
    dst0 += inc; 
  }
}

static void meta_args_check(const int max_rows_height, const int64_t row_width, const int64_t* src_rows,
                  const int64_t src_rows_count, int64_t* merged_rows, int64_t* merged_rows_count,
                  unsigned char* bitmap, int64_t* posmap, int* buff) {
  assert(src_rows_count >= 0 && src_rows_count <= max_rows_height);
  assert(row_width > 0);
  assert(src_rows != NULL);
  assert(merged_rows != NULL);
  assert(merged_rows_count != NULL);
  assert(bitmap != NULL);
  assert(posmap != NULL);
  assert(buff != NULL);
}

__attribute__ ((visibility("default")))
int allReduceMeta(const int max_rows_height, const int64_t row_width, const int64_t* src_rows,
                  const int64_t src_rows_count, int64_t* merged_rows, int64_t* merged_rows_count,
                  unsigned char* bitmap, int64_t* posmap, int* buff,
                  ncclComm_t comm, cudaStream_t stream) {
  meta_args_check(max_rows_height, row_width, src_rows, src_rows_count,
                  merged_rows, merged_rows_count, bitmap, posmap, buff);
  int device = -1;
  warpNcclCommCuDevice(comm, &device);
  assert(device != -1);
  CudaDeviceGuard guard(device);

  const int count = max_rows_height;
  const int nnz = src_rows_count;
  CUDA_CHECK(cudaMemsetAsync(static_cast<void*>(bitmap), 0, count*sizeof(unsigned char), stream));
  CUDA_CHECK(cudaMemsetAsync(static_cast<void*>(posmap), -1, count*sizeof(int64_t), stream));

  int blocks = min(DIVUP(nnz, BLOCK_SIZE), MAX_BLOCK);
  markRowsKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(bitmap, posmap, count, src_rows, nnz);

  // merge rows
  warpNcclAllReduce(bitmap, bitmap, count, ncclUint8, ncclMax, comm, stream);

  // get merged_rows and merged_rows_count from merged bitmap. buff use for thread_count
  dense2idx(merged_rows, merged_rows_count, bitmap, buff, count, stream);
  return true;
}

static void tensor_args_check(const int max_rows_height, const int64_t row_width,
            const int64_t* merged_rows, const int64_t merged_rows_count, const float* src_tensor,
            const int64_t* posmap, float* merged_tensor) {
  assert(merged_rows_count >= 0 && merged_rows_count <= max_rows_height);
  assert(row_width > 0);
  assert(merged_rows != NULL);
  assert(src_tensor != NULL);
  assert(posmap != NULL);
  assert(merged_tensor != NULL);
}

__attribute__ ((visibility("default")))
int allReduceTensor(const int max_rows_height, const int64_t row_width, const int64_t* merged_rows,
                    const int64_t merged_rows_count, const float* src_tensor, const int64_t* posmap,
                    float* merged_tensor, ncclComm_t comm, cudaStream_t stream) {
  tensor_args_check(max_rows_height, row_width, merged_rows, merged_rows_count,
                    src_tensor, posmap, merged_tensor);
  int device = -1;
  warpNcclCommCuDevice(comm, &device);
  assert(device != -1);
  CudaDeviceGuard guard(device);

  int blocks = min(static_cast<int>(merged_rows_count), MAX_BLOCK);
  // gather src val to dst
  gatherRowsValKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(merged_tensor, src_tensor, row_width,
                                            merged_rows, merged_rows_count, posmap);

  warpNcclAllReduce(merged_tensor, merged_tensor, merged_rows_count * row_width,
                    ncclFloat, ncclSum, comm, stream);
  return true;
}

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

