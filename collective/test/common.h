#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>

#include "nccl.h"

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

#define CURAND_CHECK(cmd) do {                \
     curandStatus_t error = (cmd);            \
     if (error != CURAND_STATUS_SUCCESS) {    \
         std::cout << "CuRAND failure "       \
                   << __FILE__ << ":"         \
                   << __LINE__ << " "         \
                   << std::endl;              \
         exit(EXIT_FAILURE);                  \
     }                                        \
} while (0)

void RandomizeFloat(void* dest, const int count, const int seed);
void FeedInputFloat(float * dest, const int count, const float * src, const int size);
void CheckDelta(float* dst, float* dst_test, size_t count, double* dmax, int rank, cudaStream_t stream);
