#include "sparse_comm.h"
#include "dgc.h"

#ifdef __cplusplus
extern "C"{
#endif

bool sparseAllGReduce(const void* encode, void* gatherbuff, const int nnz,
                     float* dense, const int count, ncclComm_t comm, cudaStream_t stream){
    return paddle::communication::dgc::sparseAllGReduce(encode, gatherbuff, nnz, dense, count, comm, stream);
}

bool dgc_k_select(void* encode, int k, float* input, int count, void* buff, cudaStream_t stream, float* moment ){
    return paddle::communication::dgc::k_select(encode, k, input, count, buff, stream, moment);
}

int get_dgc_buffer_size(int count){
    return paddle::communication::dgc::get_buffer_size(count);
}

int get_dgc_encode_size(int count, int k){
    return paddle::communication::dgc::get_encode_size(count,k);
}

bool is_recommend_use_dgc(int nranks, int count, int k){
    return paddle::communication::dgc::is_recommend_use_dgc(nranks, count,k);
}

/// selectRows

int allReduceMeta(const int max_rows_height, const int64_t row_width, const int64_t* src_rows,
                  const int64_t src_rows_count, int64_t* merged_rows, int64_t* merged_rows_count,
                  unsigned char* bitmap, int64_t* posmap, int* buff,
                  ncclComm_t comm, cudaStream_t stream){
    return paddle::communication::dgc::allReduceMeta(max_rows_height, row_width, src_rows, 
            src_rows_count, merged_rows, merged_rows_count,
            bitmap, posmap, buff,
            comm, stream);
}

int allReduceTensor(const int max_rows_height, const int64_t row_width, const int64_t* merged_rows,
                    const int64_t merged_rows_count, const float* src_tensor, const int64_t* posmap,
                    float* merged_tensor, ncclComm_t comm, cudaStream_t stream){
    return paddle::communication::dgc::allReduceTensor(max_rows_height, row_width, merged_rows,
            merged_rows_count, src_tensor, posmap,
            merged_tensor, comm, stream);
}
 
/// csc 
bool cscAllReduce(float* gradient, const size_t count,
                  float* ipt_chunks, const size_t chunk_width,
                  const int* ipt_idx, const int k,
                  ncclComm_t comm, cudaStream_t stream){
    return paddle::communication::dgc::cscAllReduce(gradient, count,
            ipt_chunks, chunk_width,
            ipt_idx, k,
            comm, stream);
}
 
bool calImportant(int* ipt_idx, const int k, bool* ipt_bitmap,
                  float* norms, const size_t chunk_count,
                  const float* input, const size_t count,
                  ncclComm_t comm, cudaStream_t stream){
    return paddle::communication::dgc::calImportant(ipt_idx, k, ipt_bitmap,
            norms, chunk_count,
            input, count,
            comm, stream);
}
#ifdef __cplusplus
};
#endif
