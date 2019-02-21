#include "common.h"
#include "dgc.h"
#include "threshold.h"
#include "encode.h"

#include <assert.h>

namespace paddle {
namespace communication {
namespace dgc{

#define FABS(a) ((a!=-INFINITY) ? fabs(a) : a)

#define MIN_COUNT 16384

template<typename T>
static void args_check(void* encode, void* buff, T* input, int k, int count) {
  assert(encode != NULL);
  assert(buff != NULL);
  assert(input != NULL);
  assert(k <= count); 
}

int get_buffer_size(int count) {
  return (16 + (MAX_BLOCKS + 1) * MAX_THREADS) * sizeof(float);
}

int get_encode_size(int count, int k) {
  return k * (sizeof(int) + sizeof(float));
}

bool k_select(void* encode, int k, float* input, int count, void* buff, cudaStream_t stream, float* moment) {
  args_check(encode, buff, input, k, count);

  if (count < MIN_COUNT) {
    return false;
  }
  float* threshold = static_cast<float*>(buff);
  get_threshold(threshold, input, count, k, stream);
  int* thr_cnt = static_cast<int*>(buff) + 16;
  dense2coo(encode, input, threshold, thr_cnt, count, k, stream);
  mask(encode, count, k, input, stream, moment);
  return true;
}

bool is_recommend_use_dgc(int nranks, int count, int k) {
  if (count/k < nranks*2) return false;
  if (count < 2*1024*1024) return false;
  return true;
}

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

