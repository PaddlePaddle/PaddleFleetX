#include "common.h"
#include "dgc.h"
#include "ncclwrap.h"

#include <assert.h>

namespace paddle {
namespace communication {
namespace dgc{

#define KER_CHECK(cmd) do {  \
  if (!(cmd)) { return; }    \
} while(0)

#define BLOCK_SIZE 256
#define MAX_BLOCK 32

template<typename T>
static void argsCheck(void* gatherbuff, const int nnz, T* dense, const int count) {
  assert(gatherbuff != NULL);
  assert(dense != NULL);
  assert(nnz < count);
}

template<typename T>
__global__
void sparseReduceKernel(void* encode, const int nnz, T* dense, const int count, const int nranks) {
  const int tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  const int N = BLOCK_SIZE * MAX_BLOCK;

  int* index = static_cast<int*>(encode);
  T* value = reinterpret_cast<T*>(index + nnz);
  for (int j = tid; j < nnz; j += N) {
    int idx = index[j];
    KER_CHECK((idx >= 0 && idx < count) || (idx == -1));
    if (idx == -1) continue; 
    dense[idx] += value[j];
  }
}

#ifdef DEBUG_PRINT
__global__
void checkEncodeKernel(void* encode, const int nnz, const int count) {
  __shared__ int flag;
  const int tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  const int N = BLOCK_SIZE * MAX_BLOCK;

  if (tid == 0) flag = 0;
  __syncthreads();

  int* index = static_cast<int*>(encode);
  float* value = reinterpret_cast<float*>(index + nnz);
  for (int j = tid; j < nnz; j += N) {
    int idx = index[j];
    if ( !((idx >= 0 && idx < count) || (idx == -1)) ) {
      flag |= 4;
      break;
    }
    if (idx == -1) continue;
    if (isnan(value[j])) flag |= 1;
    if (isinf(value[j])) flag |= 2;
  }

  if (tid == 0) {
    if (flag & 1) printf("===== DGC Debug: gather has nan ====");
    if (flag & 2) printf("===== DGC Debug: gather has inf ====");
    if (flag & 4) printf("===== DGC Debug: gather idx ill ====");
  }
}
#endif

template<typename T>
void sparseReduceBuff(void* gatherbuff, const int nnz, T* dense, const int count,
                      const int nranks, cudaStream_t stream) {
  const int blocks = min(DIVUP(nnz, BLOCK_SIZE), MAX_BLOCK);
  void* encode = gatherbuff;
  // Todo. cuda9 use cooperative groups for grid sync
  for (int i = 0; i < nranks; ++i) {
#ifdef DEBUG_PRINT
    checkEncodeKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(encode, nnz, count);
#endif
    sparseReduceKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(encode, nnz, dense, count, nranks);
    encode = static_cast<void*>(static_cast<char*>(encode) + (sizeof(int)+sizeof(T)) * nnz);
  }
}

__attribute__ ((visibility("default")))
bool sparseReduce(void* gatherbuff, const int nnz, float* dense, const int count,
                  const int nranks, cudaStream_t stream) {
  argsCheck(gatherbuff, nnz, dense, count);

  sparseReduceBuff(gatherbuff, nnz, dense, count, nranks, stream);
  return true;
}

__attribute__ ((visibility("default")))
bool dynloadNcclLib(void) {
  if (warpNcclSymbols() != true) {
    LOGERR("Failed dynamic load libnccl.so");
    return false;
  }
  return true;
}

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

