#include "common.h"

const int CUDA_NUM_THREADS = 512;

inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

void RandomizeFloat(void* dest, const int count, const int seed) {
    float* ptr = static_cast<float*>(dest);
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandGenerateUniform(gen, ptr, count));
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void KeFeedInputFloat(float * dest, const int count, float * src, const int size) {
    int offset = (threadIdx.x + blockDim.x * blockIdx.x) % size;

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += gridDim.x * blockDim.x) {
        dest[i] = src[offset];
        offset = (offset+1) % size;
    }
}

void FeedInputFloat(float * dest, const int count, const float * src, const int size) {
    float* g_src;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_src), size*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g_src, src, size*sizeof(float), cudaMemcpyHostToDevice));
    KeFeedInputFloat<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            dest, count, g_src, size);
    CUDA_CHECK(cudaFree(g_src));
}

template<typename T, int BSIZE>
__global__ void deltaKe(const T* A, const T* B, size_t count, double* dmax, int rank) {
  __shared__ double temp[BSIZE];
  int tid = threadIdx.x;
  double locmax = 0.0;
  for(int i=tid; i<count; i+=blockDim.x) {

    double delta = fabs((double)(A[i] - B[i]));
    if( delta > locmax ) {
      locmax = delta;
      if (delta > 0.001 && rank == 0) printf("Error at %d/%d : %f != %f, del=%lf\n", i, (int)count, (float)A[i], (float)B[i], delta);
    }
  }

  temp[tid] = locmax;
  for(int stride = BSIZE/2; stride > 1; stride>>=1) {
    __syncthreads();
    if( tid < stride )
      temp[tid] = temp[tid] > temp[tid+stride] ? temp[tid] : temp[tid+stride];
  }
  __syncthreads();
  if( threadIdx.x == 0)
    *dmax = temp[0] > temp[1] ? temp[0] : temp[1];
}

__global__ void setDelta(double* dmax, double value) {
  *dmax = value;
}

void SetDelta(double* dmax, double value) {
  setDelta<<<1, 1>>>(dmax, value);
}

void CheckDelta(float* dst, float* dst_test, size_t count, double* dmax, int rank, cudaStream_t stream) {
  deltaKe<float, 512><<<1, 512, 0, stream>>>(dst, dst_test, count, dmax, rank);
}

