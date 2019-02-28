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
#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

template<typename T>
static void argsCheck(const void* encode, void* gatherbuff, const int nnz, T* dense, const int count) {
  assert(encode != NULL);
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

template<typename T>
void sparseReduce(void* gatherbuff, const int nnz, T* dense, const int count,
                  const int nranks, cudaStream_t stream) {
  const int blocks = min(DIVUP(nnz, BLOCK_SIZE), MAX_BLOCK);
  void* encode = gatherbuff;
  // Todo. cuda9 use cooperative groups for grid sync
  for (int i = 0; i < nranks; ++i) {
    sparseReduceKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(encode, nnz, dense, count, nranks);
    encode = static_cast<void*>(static_cast<char*>(encode) + (sizeof(int)+sizeof(T)) * nnz);
  }
}

__attribute__ ((visibility("default")))
bool sparseAllGReduce(const void* encode, void* gatherbuff, const int nnz, float* dense,
                      const int count, ncclComm_t comm, cudaStream_t stream) {
  argsCheck(encode, gatherbuff, nnz, dense, count);

  warpNcclAllGather(encode, gatherbuff, nnz * (sizeof(int) + sizeof(float)), ncclChar, comm, stream);
  int nranks = 0;
  warpNcclCommCount(comm, &nranks);

  int saved_dev = -1;
  CUDA_CHECK(cudaGetDevice(&saved_dev));

  int device = -1;
  warpNcclCommCuDevice(comm, &device);
  assert(device != -1);
  CUDA_CHECK(cudaSetDevice(device));
  
  sparseReduce(gatherbuff, nnz, dense, count, nranks, stream);

  if (saved_dev != -1) CUDA_CHECK(cudaSetDevice(saved_dev));
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

