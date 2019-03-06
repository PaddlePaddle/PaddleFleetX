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

#define TIME_BENCH(item, iter, func, ret)                                  \
    {                                                                      \
        auto start = std::chrono::high_resolution_clock::now();            \
        for (int i = 0; i < iter; i++) {ret = func;}                       \
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

class CudaAlloc : public paddle::communication::dgc::GpuAllocator {
private:
  void* _ptr{NULL};
  void* _now_ptr{NULL};
  size_t _size{0};
  size_t _allocated_size{0};

public:
  CudaAlloc(size_t bytes_size) {
    void* dev_ptr = NULL;
    CUDA_CHECK(cudaMalloc(&dev_ptr, bytes_size));
    if (dev_ptr != NULL) {
      _ptr = dev_ptr;
      _now_ptr = dev_ptr;
      _size = bytes_size; 
    }
  }

  void* allocator(size_t bytes_size) {
    void* dev_ptr = NULL;
    if (_now_ptr != NULL) {
      if (_size < _allocated_size + bytes_size) {
      } else {
        dev_ptr = _now_ptr;
        _now_ptr = (void*)((char*)_now_ptr + bytes_size);
        _allocated_size += bytes_size;
      }
    }
    return dev_ptr;
  }

  ~CudaAlloc() {
    CUDA_CHECK(cudaFree(_ptr));
  }
};
    
void* AllocateCuda(int bytes) {
    void* ret;
    CUDA_CHECK(cudaMalloc(&ret, bytes));
    return ret;
}

void DeleteCuda(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

bool check_data(double* delta, const std::vector<int>& dst_rows, float* dst, float* src,
                const int row_width, const std::map<int, int>& posmap,
                int rank, ncclComm_t comm, cudaStream_t stream) {
    namespace pdgc = paddle::communication::dgc;
    bool ret = true;
    *delta = -1;

    const int nnz = dst_rows.size();
    if(nnz <= 0) return false;
    std::vector<int> dst_rows_test(nnz);
    MPICHECK(MPI_Allreduce(dst_rows.data(), dst_rows_test.data(), nnz,
             MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (dst_rows != dst_rows_test) ret = false;
//    for (int i = 0; i < nnz; ++i) {
//        printf("%d ", dst_rows_test[i]);
//    }
//    printf("\n");

    float* dst_test = (float*)AllocateCuda(nnz * row_width * sizeof(float));
    double* dmax = (double*)AllocateCuda(sizeof(double));
    for (int i = 0; i < nnz; ++i) {
        std::map<int, int>::const_iterator it = posmap.find(dst_rows[i]);
        if (it != posmap.end()) {
//           printf("it.firse=%d it.seconde=%d\n", it->first, it->second);
           CUDA_CHECK(cudaMemcpyAsync(dst_test + i*row_width, src + it->second*row_width,
                           sizeof(float)*row_width, cudaMemcpyDeviceToDevice, stream)); 
        } else {
           CUDA_CHECK(cudaMemsetAsync(dst_test + i*row_width, 0, sizeof(float)*row_width, stream));
        }
    }
//    printf("nnz=%d, row_width=%d\n", nnz, row_width);
    NCCLCHECK(ncclAllReduce(dst_test, dst_test, nnz*row_width, ncclFloat, ncclSum, comm, stream));

    if (rank == 0) CheckDelta(dst, dst_test, nnz*row_width, dmax, rank, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) {
        CUDA_CHECK(cudaMemcpy(delta, dmax, sizeof(double), cudaMemcpyDeviceToHost));
        if (*delta > 0.001 || *delta < 0) ret = false;
    }

    DeleteCuda(dst_test);
    DeleteCuda(dmax);
    return ret;
}

int  select_rows_test(const int count, const std::vector<int>& src_rows,
    const int row_width, const float* src, std::vector<int>* dst_rows,
    float** dst, paddle::communication::dgc::GpuAllocator* allocator, ncclComm_t comm, cudaStream_t stream) {
    namespace pdgc = paddle::communication::dgc;
    return pdgc::allReduceSelectedRows(count, src_rows, row_width, src, dst_rows,
                                       dst, allocator, comm, stream);
}

int runTest(int argc, char *argv[], int rank, int ranks, ncclComm_t comm) {
    namespace pdgc = paddle::communication::dgc;
    // need init before sparseAllGReduce
    pdgc::dynloadNcclLib();

    int count = 10000;
    const int row_width = 1024;
    //std::vector<int> src_rows = {0, 7, 11, 15, 20, 23, 44, 37, 56, 87};
    //std::vector<int> src_rows = {20, 23, 44, 37, 56, 87, 0, 7, 11, 15};
    std::vector<int> src_rows;
    std::vector<int> dst_rows;
    for(int i = 0; i < 300; ++i) {
        src_rows.push_back(i);
    }
    for (int i = 600; i < 10000; i += 377) {
      if (rank == 0) {
         src_rows.push_back(i-2);
      } else if (rank % 2 == 0) {
         src_rows.push_back(i);
      } else {
         src_rows.push_back(i - 5);
      }
    }

    std::map<int, int> posmap;
    for (int i = 0; i < src_rows.size(); ++i) {
        posmap[src_rows[i]] = i;
    }

    int nnz = src_rows.size();
    int bytes;
    bytes = nnz * row_width * sizeof(float);  

    const int iteration = 1;
    bool print_header = false;

    CudaAlloc aa(1024*1024*1024);  // 1GB
    pdgc::GpuAllocator* cualloc = &aa;
    // allocate src
    float* src = static_cast<float*>(AllocateCuda(bytes));
    float* dst;
    // randomize input data
    FeedInputFloat(src, nnz * row_width, g_chunk, CHUNKSIZE);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float* tmp = static_cast<float*>(AllocateCuda(1024*1024*sizeof(float)));
    for (int i = 0; i < 5; i++) {
      NCCLCHECK(ncclAllReduce(tmp, tmp, 1024*1024, ncclFloat, ncclSum, comm, stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    int is_ok = 0;
    float time = 0;
    TIME_BENCH(time, iteration,
        select_rows_test(count, src_rows, row_width, src,
                         &dst_rows, &dst, cualloc, comm, stream), is_ok);
    std::string ret;
    double delta;
    if (is_ok) {
        ret = check_data(&delta, dst_rows, dst, src, row_width, posmap, rank, comm, stream) ? "AC" : "WA";
        //ret = "AC";
    } else {
        ret = "WA";
    }
    if (rank == 0) {
        if (print_header) {
            std::string suffix = " us";
            std::cout << std::right;
            std::cout << std::setw(13) << "elements";
            std::cout << std::setw(13) << "dtype";
            std::cout << std::setw(13) << ("time" + suffix);
            std::cout << std::setw(13) << "check";
            std::cout << std::setw(13) << "delta";
            std::cout << std::endl;
        }
        std::cout << std::setw(13) << count;
        std::cout << std::setw(13) << "float";
        std::cout << std::setw(13) << time;
        std::cout << std::setw(13) << ret;
        std::cout << std::setw(13) << delta;
        std::cout << std::endl;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    DeleteCuda(src);
    DeleteCuda(tmp);
}

int main(int argc, char *argv[]) {
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
