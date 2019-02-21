#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <memory.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>
#include <unistd.h>

#include "nccl.h"
#include "mpi.h"
#include "dgc.h"

#include "common.h"
#include "data.h"

#define LOGERR(format, args...) (fprintf(stderr, "[%s:%d:%s] " format "\n",\
                                         __FILE__, __LINE__, __FUNCTION__, ##args))

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define TIME_BENCH(item, iter, func)                                       \
    {                                                                      \
        auto start = std::chrono::high_resolution_clock::now();            \
        for (int i = 0; i < iter; i++) {func;}                             \
        CUDA_CHECK(cudaDeviceSynchronize());                               \
        auto end = std::chrono::high_resolution_clock::now();              \
        item = std::chrono::nanoseconds(end - start).count() / 1000./iter; \
    }

extern float data[CHUNKSIZE];

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

bool compare(float a, float b) {
    return fabs(a)>fabs(b);
}

void* AllocateCuda(int bytes) {
    void* ret;
    CUDA_CHECK(cudaMalloc(&ret, bytes));
    return ret;
}

void DeleteCuda(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

bool check_data(int ranks, float * data, int count, float * data_out, 
                    void * encode, void * encode2, 
                    void * buffer, float * moment, int k, 
                    float* kth,  float* rkth = nullptr) {
    namespace pdgc = paddle::communication::dgc;

    bool ret = true;
    int bytes, kbytes, buffbytes;
    bytes = count * sizeof(float);
    kbytes = pdgc::get_encode_size(count, k);
    buffbytes= pdgc::get_buffer_size(count);
    float * h_data = static_cast<float*>(malloc(bytes));
    float * h_data_o = static_cast<float*>(malloc(bytes));
    void * h_encode = malloc(kbytes);
    void * h_encode2 = malloc(kbytes);
    void * h_buffer = malloc(buffbytes);
    float* h_moment = static_cast<float*>(malloc(bytes));

    CUDA_CHECK(cudaMemcpy(h_data, data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_data_o, data_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_encode, encode, kbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_encode2, encode2, kbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_buffer, buffer, buffbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_moment, moment, bytes, cudaMemcpyDeviceToHost));

    *kth = (static_cast<float*>(h_buffer))[0];
    if (rkth != nullptr) {
        std::partial_sort(h_data, h_data + k, h_data + count, compare);
        *rkth = h_data[k-1];
    }
    int* index1 = static_cast<int*>(h_encode);
    int* index2 = static_cast<int*>(h_encode2);
    float* value1 = static_cast<float*>(h_encode) + k;
    float* value2 = static_cast<float*>(h_encode2) + k;
    for (int i = 0; i < k; i++) {
        if (index1[i] != index2[i] || value1[i] - value2[i] > 0.0000001) {
            ret = false;
            break;
        }
        if (index1[i] == -1) {
            break;
        }
//        if (h_moment[index1[i]] != 0 || h_data_o[index1[i]] != 0) {
        if (fabs(value1[i] * ranks - h_data_o[index1[i]]) > 0.000001 * ranks) {
            printf("i=%d, index=%d, encode=%f, output=%f, multi_ans=%f\n", i, index1[i], value1[i], h_data_o[index1[i]], value1[i]*ranks);
            ret = false;
            break;
        }
    }

    free(h_data);
    free(h_data_o);
    free(h_encode);
    free(h_encode2);
    free(h_buffer);
    free(h_moment);
    return ret;
}

void cal_com_test(float* data, int count, void* encode, void* buffer, int k,
                  cudaStream_t stream, float* moment,
                  void* gather_buff, ncclComm_t comm) {
    namespace pdgc = paddle::communication::dgc;
    pdgc::k_select(encode, k, data, count, buffer, stream, moment);
//    k_select_bucket(data, count, encode , buffer, k, stream, moment);
    pdgc::sparseAllGReduce((const void*)encode, gather_buff, k,
                                data, count, comm, stream);
}


int runTest(int argc, char *argv[], int rank, int ranks, ncclComm_t comm) {
    namespace pdgc = paddle::communication::dgc;
    // need init before sparseAllGReduce
    pdgc::dynloadNcclLib();

    int count = atoi(argv[1]);
    int k = atoi(argv[2]);

//    if (rank == 0) { 
//        printf("is recommend use dgc %d\n", pdgc::is_recommend_use_dgc(ranks, count, k));
//    }

    int bytes, kbytes, bbytes;
    bytes = count * sizeof(float);
    kbytes = pdgc::get_encode_size(count, k);
    bbytes = pdgc::get_buffer_size(count);
    const int iteration = 1;
    bool print_header = false;
    if (argc == 4) {
        if (atoi(argv[3]) == 1) print_header = true;
        else print_header = false;
    }

    // allocate buffers
    float * data   = static_cast<float*>(AllocateCuda(bytes));
    float * moment = static_cast<float*>(AllocateCuda(bytes));
    void *  buffer = AllocateCuda(bbytes);
    void *  encode = AllocateCuda(kbytes);
    void *  encode2= AllocateCuda(kbytes);
    float * data1  = static_cast<float*>(AllocateCuda(bytes));
    float * data2  = static_cast<float*>(AllocateCuda(bytes));

    void *  gather_buff = AllocateCuda(kbytes * ranks);
    void *  gather_buff1 = AllocateCuda(kbytes * ranks);

    // randomize input data
    FeedInputFloat(data, count, g_chunk, CHUNKSIZE);
    FeedInputFloat(moment, count, g_chunk, CHUNKSIZE);
    FeedInputFloat(data1, count, g_chunk, CHUNKSIZE);
    FeedInputFloat(data2, count, g_chunk, CHUNKSIZE);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < 10; i++) {
      ncclAllGather((const void*)encode, gather_buff, kbytes, ncclChar, comm, stream);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float time = 0;
    float kth = -1;
    float rkth = -1;
    TIME_BENCH(time, iteration,
        cal_com_test(data1, count, encode, buffer, k, stream, moment,
                     gather_buff, comm));
    TIME_BENCH(time, 1, 
        cal_com_test(data2, count, encode2, buffer, k, stream, moment,
                     gather_buff1, comm));
    std::string ret;
    ret = check_data(ranks, data, count, data1, encode, encode2, buffer, moment,
                                     k, &kth, &rkth) ? "OK" : "Err";
    if (rank % 8 == 0) {
        if (print_header) {
            std::string suffix = " us";
            std::cout << std::right;
            std::cout << std::setw(13) << "elements";
            std::cout << std::setw(13) << "topk";
            std::cout << std::setw(13) << "dtype";
            std::cout << std::setw(13) << ("time" + suffix);
            std::cout << std::setw(13) << "check";
            std::cout << std::setw(13) << "kth";
            std::cout << std::setw(13) << "rkth";
            std::cout << std::endl;
        }
        std::cout << std::setw(13) << count;
        std::cout << std::setw(13) << k;
        std::cout << std::setw(13) << "float";
        std::cout << std::setw(13) << time;
        std::cout << std::setw(13) << ret;
        std::cout << std::setw(13) << kth;
        std::cout << std::setw(13) << rkth;
        std::cout << std::endl;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    DeleteCuda(data);
    DeleteCuda(data1);
    DeleteCuda(data2);
    DeleteCuda(encode);
    DeleteCuda(encode2);
    DeleteCuda(buffer);
    DeleteCuda(moment);
    DeleteCuda(gather_buff);
    DeleteCuda(gather_buff1);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        LOGERR("%s count topk", argv[0]);
        exit(-1);
    }

    int my_global_rank = 0;
    int ranks = 0;
    int local_rank = 0;

    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_global_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    // hash address of each rank as a uint64
    uint64_t host_hash[ranks];
    char hostname[1024];
    getHostName(hostname, 1024);
    host_hash[my_global_rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                           host_hash, sizeof(uint64_t),
                           MPI_BYTE, MPI_COMM_WORLD));
    // init nccl
    for (int i = 0; i < ranks; ++i) {
        if (i == my_global_rank) {
            break;
        }
        if (host_hash[i] == host_hash[my_global_rank]) {
            local_rank++;
        }
    }
    ncclUniqueId id;
    ncclComm_t comm;
    if (my_global_rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0,
                       MPI_COMM_WORLD));
    CUDA_CHECK(cudaSetDevice(local_rank));
    NCCLCHECK(ncclCommInitRank(&comm, ranks, id, my_global_rank));

    runTest(argc, argv, my_global_rank, ranks, comm);
    
    NCCLCHECK(ncclCommDestroy(comm));
    MPICHECK(MPI_Finalize());
    return 0;
}
