#ifndef THRESHOLD_H_
#define THRESHOLD_H_

#include "common.h"

namespace paddle {
namespace communication {
namespace dgc{

void get_threshold(float* threshold, float* input, int count, int k, cudaStream_t stream);

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

#endif
