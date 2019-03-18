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

__attribute__ ((visibility("default")))
int get_buffer_size(int count) {
  return (16 + (MAX_BLOCKS + 1) * MAX_THREADS) * sizeof(float);
}

__attribute__ ((visibility("default")))
int get_encode_size(int count, int k) {
  return k * (sizeof(int) + sizeof(float));
}

__attribute__ ((visibility("default")))
bool k_select(void* encode, int k, float* input, int count, void* buff, cudaStream_t stream, float* moment) {
  args_check(encode, buff, input, k, count);

  /// ptr check
  {
    int saved_dev = -1;
    CUDA_CHECK(cudaGetDevice(&saved_dev));
    devptr_check((const void*)encode, saved_dev, "encode");
    devptr_check((const void*)input, saved_dev, "input");
    devptr_check((const void*)buff, saved_dev, "buff");
  }

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

__attribute__ ((visibility("default")))
bool is_recommend_use_dgc(int nranks, int count, int k) {
  if (count/k < nranks*2) return false;
  if (count < 2*1024*1024) return false;
  return true;
}

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

