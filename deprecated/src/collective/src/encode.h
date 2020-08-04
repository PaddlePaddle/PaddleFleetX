#ifndef ENCODE_H_
#define ENCODE_H_

#include "common.h"

namespace paddle {
namespace communication {
namespace dgc{

void dense2coo(void* encode, float * input, float* threshold, int* thr_cnt, int count, int k, cudaStream_t stream);
void mask(void* encode, int count, int k, float* input, cudaStream_t stream, float* moment = NULL);

/// for selectRows
void dense2idx(int64_t* index, int64_t* nnz, unsigned char* input, int* thr_cnt, int count, cudaStream_t stream);

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

#endif
