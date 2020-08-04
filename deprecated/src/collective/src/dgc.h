#ifndef DGC_H
#define DGC_H

#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>
#include "nccl.h"

namespace paddle {
namespace communication {
namespace dgc {

/**
 * dynloadNcclLib
 * Dynamic load nccl lib, need init before sparseAllGReduce.
 */
bool dynloadNcclLib(void);

/**
 * sparseReduce
 *
 * Decode and reduce the gather encode to dense array.
 *
 * @param gatherbuff - starting address of gather encode, gather encode should gather from encode use ncclAllGather, and have a size at least nranks*encode_size.
 * @param nnz - the count of non-zero elements in sparse array.
 * @param dense - starting address of dense array.
 * @param count - the count of dense array.
 * @param nranks - the count of ranks.
 * @param stream - cuda stream.
 * @return
 */
bool sparseReduce(void* gather_encode, const int nnz, float* dense,
                  const int count, const int nranks, cudaStream_t stream);

bool k_select(void* encode, int k, float* input, int count, void* buff, cudaStream_t stream, float* moment = NULL);

int get_buffer_size(int count); 

int get_encode_size(int count, int k);

bool is_recommend_use_dgc(int nranks, int count, int k);

/// selectRows

int allReduceMeta(const int max_rows_height, const int64_t row_width, const int64_t* src_rows,
                  const int64_t src_rows_count, int64_t* merged_rows, int64_t* merged_rows_count,
                  unsigned char* bitmap, int64_t* posmap, int* buff,
                  ncclComm_t comm, cudaStream_t stream);

int allReduceTensor(const int max_rows_height, const int64_t row_width, const int64_t* merged_rows,
                    const int64_t merged_rows_count, const float* src_tensor, const int64_t* posmap,
                    float* merged_tensor, ncclComm_t comm, cudaStream_t stream);
 
/// csc 

bool cscAllReduce(float* gradient, const size_t count,
                  float* ipt_chunks, const size_t chunk_width,
                  const int* ipt_idx, const int k,
                  ncclComm_t comm, cudaStream_t stream);
 
bool calImportant(int* ipt_idx, const int k, bool* ipt_bitmap,
                  float* norms, const size_t chunk_count,
                  const float* input, const size_t count,
                  ncclComm_t comm, cudaStream_t stream);

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

#endif
