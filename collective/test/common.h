#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <unistd.h>

#include "nccl.h"

#define LOGERR(format, args...) (fprintf(stderr, "[%s:%d:%s] " format "\n",\
                                         __FILE__, __LINE__, __FUNCTION__, ##args))

#define CUDA_CHECK(cmd) do {                  \
  cudaError_t e = cmd;                      \
  if (e != cudaSuccess) {                   \
    std::cout << "Cuda failure "          \
              << __FILE__ << ":"          \
              << __LINE__ << " "          \
              << cudaGetErrorString(e)    \
              << std::endl;               \
    exit(EXIT_FAILURE);                   \
  }                                         \
} while (0)

#define CURAND_CHECK(cmd) do {             \
  curandStatus_t error = (cmd);            \
  if (error != CURAND_STATUS_SUCCESS) {    \
      std::cout << "CuRAND failure "       \
                << __FILE__ << ":"         \
                << __LINE__ << " "         \
                << std::endl;              \
      exit(EXIT_FAILURE);                  \
  }                                        \
} while (0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define TIME_BENCH(item, iter, func, ret) do {                             \
  auto start = std::chrono::high_resolution_clock::now();            \
  for (int i = 0; i < iter; i++) {ret = func;}                       \
  CUDA_CHECK(cudaDeviceSynchronize());                               \
  auto end = std::chrono::high_resolution_clock::now();              \
  item = std::chrono::nanoseconds(end - start).count() / 1000./iter; \
} while(0)


void RandomizeFloat(void* dest, const int count, const int seed);
void FeedInputFloat(float * dest, const int count, const float * src, const int size);
void CheckDelta(float* dst, float* dst_test, size_t count, double* dmax, int rank, cudaStream_t stream);
void SetDelta(double* dmax, double value);

static void getHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

static uint64_t getHostHash(const char *string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void* AllocateCuda(int bytes) {
    void* ret;
    CUDA_CHECK(cudaMalloc(&ret, bytes));
    return ret;
}

static void DeleteCuda(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}
