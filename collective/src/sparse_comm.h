#ifndef __SPARSE_COMM_H__

#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>
#include "nccl.h"

#ifdef __cplusplus
extern "C"{
#endif

bool sparseAllGReduce(const void* encode, void* gatherbuff, const int nnz,
                     float* dense, const int count, ncclComm_t comm, cudaStream_t stream);

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
#ifdef __cplusplus
};
#endif

#endif

