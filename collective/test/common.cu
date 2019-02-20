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
