#include "common.h"
#include "threshold.h"

namespace paddle {
namespace communication {
namespace dgc{

#define FABS(a) ((a!=-INFINITY) ? fabs(a) : a)
#define BLOCKSIZE 1024
#define MAXLENGTH_FOR_TOPK 5

static void getNumBlocksAndThreads(float n, int &blocks, int &threads) {
  threads = BLOCKSIZE;
  blocks = min(16, static_cast<int>(n / 2));
}

template <typename T>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, int id) : v(value), id(id) {}

  __device__ __forceinline__ void set(T value, int id) {
    v = value;
    id = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T>& in) {
    v = in.v;
    id = in.id;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return (FABS(v) < FABS(value));
  }

  __device__ __forceinline__ bool operator<(const Pair<T>& in) const {
    return (FABS(v) < FABS(in.v)) || ((FABS(v) == FABS(in.v)) && (id > in.id));
  }

  __device__ __forceinline__ bool operator>(const Pair<T>& in) const {
    return (FABS(v) > FABS(in.v)) || ((FABS(v) == FABS(in.v)) && (id < in.id));
  }

  T v;
  int id;
};

template <typename T>
__device__ __forceinline__ void AddTo(Pair<T> topk[], const Pair<T>& p,
                                      int beam_size) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (topk[k] < p) {
      topk[k + 1] = topk[k];
    } else {
      topk[k + 1] = p;
      return;
    }
  }
  topk[0] = p;
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      AddTo<T>(topk, tmp, beam_size);
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, const Pair<T>& max,
                                        int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      if (tmp < max) {
        AddTo<T>(topk, tmp, beam_size);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[], int* beam,
                                              int beam_size, const T* src,
                                              bool* firstStep, bool* is_empty,
                                              Pair<T>* max, int dim,
                                              const int tid) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetTopK<T, BlockSize>(topk, src, tid, dim, length);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - (*beam)) {
          topk[k] = topk[k + *beam];
        } else {
          topk[k].set(-INFINITY, -1);
        }
      }
      if (!(*is_empty)) {
        GetTopK<T, BlockSize>(topk + MaxLength - *beam, src, tid, dim, *max,
                              length);
      }
    }

    *max = topk[MaxLength - 1];
    if ((*max).v == -1) *is_empty = true;
    *beam = 0;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ T BlockReduce(Pair<T>* sh_topk, int* maxid,
                                         Pair<T> topk[],
                                         int* beam, int* k,
                                         const int tid, const int warp,
                                         int** topIds = NULL,
                                         bool* important = NULL) {
  T ret = -INFINITY;
  while (true) {
    __syncthreads();
    if (tid < BlockSize / 2) {
      if (sh_topk[tid] < sh_topk[tid + BlockSize / 2]) {
        maxid[tid] = tid + BlockSize / 2;
      } else {
        maxid[tid] = tid;
      }
    }
    __syncthreads();
    for (int stride = BlockSize / 4; stride > 0; stride = stride / 2) {
      if (tid < stride) {
        if (sh_topk[maxid[tid]] < sh_topk[maxid[tid + stride]]) {
          maxid[tid] = maxid[tid + stride];
        }
      }
      __syncthreads();
    }
    __syncthreads();

    if (tid == 0 && topIds != NULL && important != NULL) {
      **topIds = sh_topk[maxid[0]].id;
      (*topIds)++;
      important[sh_topk[maxid[0]].id] = true;
    }
    if (tid == maxid[0]) (*beam)++;
    if (--(*k) == 0) { 
      if (tid == 0) {
        ret = sh_topk[maxid[0]].v;
      }
      break;
    }
    __syncthreads();

    if (tid == maxid[0]) {
      if (*beam < MaxLength) {
        sh_topk[tid] = topk[*beam];
      }
    }

    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, true);

    if (maxid[0] / 32 == warp) {
      if (CudaShuffleSync(mask, *beam, (maxid[0]) % 32, 32) ==
          MaxLength)
        break;
    }
  }
  return ret;
}

template <typename T, int MaxLength, int BlockSize>
__global__ void KeGetTopKImportantIdx(const T* src, int lds, int dim, int k, int* index, bool* important) {
  __shared__ Pair<T> sh_topk[BlockSize];
  __shared__ int maxid[BlockSize/2];
  const int warp = threadIdx.x / 32;

  Pair<T> topk[MaxLength];
  int beam = MaxLength;
  Pair<T> max;
  bool is_empty = false;
  bool firststep = true;

  for (int k = 0; k < MaxLength; k++) {
    topk[k].set(-INFINITY, -1);
  }
  while (k) {
    ThreadGetTopK<T, MaxLength, BlockSize>(topk, &beam, k,
                                           src + blockIdx.x*lds, &firststep,
                                           &is_empty, &max, dim, threadIdx.x);
    sh_topk[threadIdx.x] = topk[0];
    T temp = BlockReduce<T, MaxLength, BlockSize>(sh_topk, maxid, topk, &beam,
                                         &k, threadIdx.x, warp, &index, important);
  }
}

template <typename T, int MaxLength, int BlockSize>
__global__ void KeGetSampleTopK(T* output, const T* src, int lds, int dim, int k) {
  __shared__ Pair<T> sh_topk[BlockSize];
  __shared__ int maxid[BlockSize / 2];
  const int warp = threadIdx.x / 32;
  T kth = -INFINITY;

  Pair<T> topk[MaxLength];
  int beam = MaxLength;
  Pair<T> max;
  bool is_empty = false;
  bool firststep = true;

  for (int k = 0; k < MaxLength; k++) {
    topk[k].set(-INFINITY, -1);
  }
  while (k) {
    ThreadGetTopK<T, MaxLength, BlockSize>(topk, &beam, k,
                                           src + blockIdx.x * lds, &firststep,
                                           &is_empty, &max, dim, threadIdx.x);

    sh_topk[threadIdx.x] = topk[0];
    T temp = BlockReduce<T, MaxLength, BlockSize>(sh_topk, maxid, topk, &beam,
                                         &k, threadIdx.x, warp);
    if (temp != -INFINITY) {
      kth = temp;
    }
  }
  if (kth != -INFINITY) {
    output[blockIdx.x] = kth;
  }
}

template <typename T, int BlockSize>
__global__ void KeGetTotalTopk(volatile T* data, int n) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  T res = 0;
  if (bid == 0 && tid == 0) {
    for (int i = 0; i < gridDim.x; i++) {
      res += FABS(data[i]);
    }
  }
  __syncthreads();
  if (bid == 0 && tid == 0) {
    data[0] = res / n;
  }
}

void get_ipt_idx(int* index, const int k, bool* important, const float* norms, const int chunks, cudaStream_t stream) {
  const int MaxLength = 5;
  const int BlockSize = 512;
  int blocks = 1;
  KeGetTopKImportantIdx<float, MaxLength, BlockSize><<<blocks, BlockSize, 0, stream>>>(
                                 norms, chunks, chunks, k, index, important);
}

void get_threshold(float* threshold, float* input, int count, int k, cudaStream_t stream) {
  int sampk = min(32, k/2);
  const float sample_prop = static_cast<float>(k) / sampk;
  int sampdim = count / sample_prop;

  int threads = -1; 
  int blocks = -1;
  getNumBlocksAndThreads(sample_prop, blocks, threads);
  int lds = count / blocks;

  KeGetSampleTopK<float, MAXLENGTH_FOR_TOPK, BLOCKSIZE><<<blocks, threads, 0, stream>>>(threshold, input, lds, sampdim, sampk);
  KeGetTotalTopk<float, BLOCKSIZE><<<blocks, threads, 0, stream>>>(threshold, blocks);
}

}  // namespace dgc
}  // namespace communication
}  // namespace paddle
