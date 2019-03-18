#ifndef THRESHOLD_H_
#define THRESHOLD_H_

#include "common.h"

namespace paddle {
namespace communication {
namespace dgc{

void get_threshold(float* threshold, float* input, int count, int k, cudaStream_t stream);

// for csc
void get_ipt_idx(int* index, const int k, bool* important, const float* norms, const int chunks, cudaStream_t stream);

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

#endif
