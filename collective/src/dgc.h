#ifndef DGC_H
#define DGC_H

#include <cuda_runtime.h>
#include "nccl.h"
#include <vector>

namespace paddle {
namespace communication {
namespace dgc {

/**
 * dynloadNcclLib
 * Dynamic load nccl lib, need init before sparseAllGReduce.
 */
bool dynloadNcclLib(void);

/**
 * sparseAllGReduce
 *
 * Gather encode of sparse array from all devices by ncclAllGather, then decode and reduce with dense array.
 *
 * @param encode - starting address of encode, data layout should be SOA {idx[@nnz], val[@nnz]}.
 * @param gatherbuff - starting address of gatherbuff, should have a size at least nranks*encode_size.
 * @param nnz - the count of non-zero elements in sparse array.
 * @param dense - starting address of dense array.
 * @param count - the count of dense array.
 * @param comm - nccl communicator
 * @param stream - cuda stream
 * @return
 */
bool sparseAllGReduce(const void* encode, void* gatherbuff, const int nnz,
                     float* dense, const int count, ncclComm_t comm, cudaStream_t stream);

bool k_select(void* encode, int k, float* input, int count, void* buff, cudaStream_t stream, float* moment = NULL);

int get_buffer_size(int count); 

int get_encode_size(int count, int k);

bool is_recommend_use_dgc(int nranks, int count, int k);

/// selectRows+CSC

class GpuAllocator {
public:
  virtual void* allocator(size_t bytes_size) = 0;
};

int allReduceSelectedRows(const int max_rows_height, const std::vector<int>& src_rows,
    const int row_width, const float* src_tensor, std::vector<int>* dst_rows,
    float** dst_tensor, GpuAllocator* allocator, ncclComm_t comm, cudaStream_t stream);

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

#endif
