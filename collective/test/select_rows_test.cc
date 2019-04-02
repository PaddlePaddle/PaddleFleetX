#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <memory.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>
#include <unistd.h>
#include <assert.h>

#include "nccl.h"
#include "mpi.h"
#include "dgc.h"

#include "common.h"
#include "data.h"

extern float data[CHUNKSIZE];

class CudaPreAlloc {
private:
  void* _ptr{NULL};
  void* _now_ptr{NULL};
  size_t _size{0};
  size_t _allocated_size{0};

public:
  CudaPreAlloc(size_t bytes_size) {
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

  void reset() {
    if (_ptr != NULL) {
      _now_ptr = _ptr;
      _allocated_size = 0; 
    }
  }

  ~CudaPreAlloc() {
    if (_ptr != NULL) CUDA_CHECK(cudaFree(_ptr));
  }
};
    
void printHeader(bool is_print) {
    if (!is_print) return;
    std::string suffix = " us";
    std::cout << std::right;
    std::cout << std::setw(13) << "elements";
    std::cout << std::setw(13) << "dtype";
    std::cout << std::setw(13) << ("time" + suffix);
    std::cout << std::setw(13) << "check";
    std::cout << std::setw(13) << "delta";
    std::cout << std::endl;
}

void printOut(int count, float time, std::string& ret, double delta) {
    std::cout << std::setw(13) << count;
    std::cout << std::setw(13) << "float";
    std::cout << std::setw(13) << time;
    std::cout << std::setw(13) << ret;
    std::cout << std::setw(13) << delta;
    std::cout << std::endl;
}

typedef struct {
    int count;                     //0
    int64_t row_width;             //1
    int64_t* src_rows;             //2
    int64_t src_rows_count;        //3
    int64_t* merged_rows;          //4
    int64_t* merged_rows_count;    //5
    unsigned char* bitmap;         //6
    int64_t* posmap;               //7
    int* buff;                     //8
    float* src_tensor;             //9
    float* merged_tensor;          //10
} testArgs;

void feedSrcRows(std::vector<int64_t>& src_rows, int rank) {
    for (int i = 100; i < 3000; ++i) src_rows.push_back(i);
    if (rank % 3 == 0) { 
      for (int i = 3000; i < 3100; ++i) src_rows.push_back(i);
    }
    for (int i = 3300; i < 100000; i += 377) {
      if (rank == 0) {
        src_rows.push_back(i-2);
      } else if (rank % 2 == 0) {
        src_rows.push_back(i);
      } else {
        src_rows.push_back(i-5);
      }
    }
}

testArgs initTestArgs(int rank) {
    namespace pdgc = paddle::communication::dgc;
    /// 0 1
    int count = 100000;
    const int64_t row_width = 1024;

    std::vector<int64_t> c_src_rows = {20, 23, 44, 37, 56, 87, 0, 7, 11, 15};
    // Todo. random init srcRows with count
    feedSrcRows(c_src_rows, rank); 

    /// 2 3
    int64_t* src_rows = static_cast<int64_t*>(AllocateCuda(sizeof(int64_t)*count)); //count
    int64_t src_rows_count = c_src_rows.size();
    CUDA_CHECK(cudaMemcpy(src_rows, c_src_rows.data(),
               sizeof(int64_t)*src_rows_count, cudaMemcpyHostToDevice));

    /// 4 5
    int64_t* merged_rows = static_cast<int64_t*>(AllocateCuda(sizeof(int64_t)*count)); //count
    int64_t* merged_rows_count = static_cast<int64_t*>(AllocateCuda(sizeof(int64_t))); //1

    /// 6 7 8
    unsigned char* bitmap = static_cast<unsigned char*>(
                              AllocateCuda(sizeof(unsigned char)*count)); //count
    int64_t* posmap = static_cast<int64_t*>(AllocateCuda(sizeof(int64_t)*count)); //count
    int* buff = static_cast<int*>(AllocateCuda(pdgc::get_buffer_size(count))); //buffsize

    // 9
    float* src_tensor = static_cast<float*>(
                          AllocateCuda(sizeof(float)*src_rows_count*row_width)); // nnz*row_width
    float* merged_tensor = NULL;    

    // randomize input data
    FeedInputFloat(src_tensor, src_rows_count * row_width, g_chunk, CHUNKSIZE);

    testArgs args = {
        .count             = count            ,
        .row_width         = row_width        ,
        .src_rows          = src_rows         ,
        .src_rows_count    = src_rows_count   ,
        .merged_rows       = merged_rows      ,
        .merged_rows_count = merged_rows_count,
        .bitmap            = bitmap           ,
        .posmap            = posmap           ,
        .buff              = buff             ,
        .src_tensor        = src_tensor       ,
        .merged_tensor     = merged_tensor    ,
    };
    return args;
}

void freeTestArgs(testArgs args) {
    DeleteCuda(args.src_rows);
    DeleteCuda(args.merged_rows);
    DeleteCuda(args.merged_rows_count);
    DeleteCuda(args.bitmap);
    DeleteCuda(args.posmap);
    DeleteCuda(args.buff);
    DeleteCuda(args.src_tensor);
}

bool check_merged_rows(const int count, std::vector<int64_t>& src_rows,
                       std::vector<int64_t>& dst_rows, int64_t nnz) {
    // test dst_rows is equal in all cards
    std::vector<int64_t> dst_rows_test(nnz);
    MPICHECK(MPI_Allreduce(dst_rows.data(), dst_rows_test.data(), nnz,
             MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD));
    if (dst_rows != dst_rows_test) {
      LOGERR("merged_rows is not equal in all cards");
      return false;
    }

    // test dst_rows is right
    unsigned char* bitmap = static_cast<unsigned char*>(malloc(sizeof(unsigned char)*count));
    memset(bitmap, 0, sizeof(unsigned char)*count);
    for (int i = 0; i < src_rows.size(); ++i) {
      bitmap[src_rows[i]] = 1; 
    }
    MPICHECK(MPI_Allreduce(MPI_IN_PLACE, bitmap, count,
             MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD));
    for (int i = 0; i < dst_rows.size(); ++i) {
      if (bitmap[dst_rows[i]] == 0) {
        LOGERR("bitmap in dst_rows[%d]=%lld should be true", i, dst_rows[i]);
        return false;
      }
      bitmap[dst_rows[i]] = 0;
    }
    for (int i = 0; i < count; ++i) {
      if(bitmap[i] != 0) {
        LOGERR("bitmap in [%d] should be false", i);
        return false;
      }
    }
    return true;
}

bool check_dst_tensor(double* delta, testArgs* args, std::vector<int64_t>&src_rows,
                      std::vector<int64_t>& dst_rows, int64_t nnz,
                      int rank, ncclComm_t comm, cudaStream_t stream) {
    int64_t row_width = args->row_width;
    float* dst_test = (float*)AllocateCuda(nnz * row_width * sizeof(float));
    double* dmax = (double*)AllocateCuda(sizeof(double));

    std::map<int64_t, int64_t> posmap;
    for (int i = 0; i < src_rows.size(); ++i) {
        posmap[src_rows[i]] = i; 
    }

    for (int i = 0; i < nnz; ++i) {
        std::map<int64_t, int64_t>::const_iterator it = posmap.find(dst_rows[i]);
        if (it != posmap.end()) {
           CUDA_CHECK(cudaMemcpyAsync(dst_test + i*row_width, args->src_tensor + it->second*row_width,
                           sizeof(float)*row_width, cudaMemcpyDeviceToDevice, stream)); 
        } else {
           CUDA_CHECK(cudaMemsetAsync(dst_test + i*row_width, 0, sizeof(float)*row_width, stream));
        }
    }
    NCCLCHECK(ncclAllReduce(dst_test, dst_test, nnz*row_width, ncclFloat, ncclSum, comm, stream));

    CheckDelta(args->merged_tensor, dst_test, nnz*row_width, dmax, rank, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    *delta = -1;
    CUDA_CHECK(cudaMemcpy(delta, dmax, sizeof(double), cudaMemcpyDeviceToHost));

    DeleteCuda(dst_test);
    DeleteCuda(dmax);

    if (*delta > 0.001 || *delta < 0) return false;
    return true;
}

bool check_data(double* delta, testArgs* args, int rank, ncclComm_t comm, cudaStream_t stream) {
    int64_t nnz = -1;
    CUDA_CHECK(cudaMemcpy(&nnz, args->merged_rows_count,
                          sizeof(int64_t), cudaMemcpyDeviceToHost));
    if(nnz <= 0) {
      LOGERR("merged_rows_count wrong");
      return false;
    }

    std::vector<int64_t> src_rows(args->src_rows_count);
    std::vector<int64_t> dst_rows(nnz);
    CUDA_CHECK(cudaMemcpy(src_rows.data(), args->src_rows,
                          sizeof(int64_t)*args->src_rows_count, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst_rows.data(), args->merged_rows,
                          sizeof(int64_t)*nnz, cudaMemcpyDeviceToHost));

    if(!check_merged_rows(args->count, src_rows, dst_rows, nnz)) return false;
    return check_dst_tensor(delta, args, src_rows, dst_rows, nnz, rank, comm, stream);
}

// select_rows time test
int timeTest(testArgs* args, CudaPreAlloc* pa, ncclComm_t comm, cudaStream_t stream) {
    namespace pdgc = paddle::communication::dgc;

    const int      count             = args->count            ;
    const int64_t  row_width         = args->row_width        ;
    const int64_t* src_rows          = args->src_rows         ;
    const int64_t  src_rows_count    = args->src_rows_count   ;
    int64_t*       merged_rows       = args->merged_rows      ;
    int64_t*       merged_rows_count = args->merged_rows_count;
    unsigned char* bitmap            = args->bitmap           ;
    int64_t*       posmap            = args->posmap           ;
    int*           buff              = args->buff             ;
    const float*   src_tensor        = args->src_tensor       ;
    float*         merged_tensor     = args->merged_tensor    ;

    pdgc::allReduceMeta(count, row_width, src_rows, src_rows_count, merged_rows, merged_rows_count,
                        bitmap, posmap, buff, comm, stream);

    int64_t c_merged_rows_count = -1;
    CUDA_CHECK(cudaMemcpy(&c_merged_rows_count, merged_rows_count,
                          sizeof(int64_t), cudaMemcpyDeviceToHost));
    assert(c_merged_rows_count != -1);
    if (merged_tensor == NULL) {
        merged_tensor = static_cast<float*>(pa->allocator(c_merged_rows_count *
                                                    row_width * sizeof(float)));
        args->merged_tensor = merged_tensor;
    }
    pdgc::allReduceTensor(count, row_width, merged_rows, c_merged_rows_count, src_tensor, posmap,
                          merged_tensor, comm, stream);
    return true;
}

int runTest(int argc, char *argv[], int rank, int ranks, ncclComm_t comm) {
    namespace pdgc = paddle::communication::dgc;
    // need init before sparseAllGReduce
    pdgc::dynloadNcclLib();

    CudaPreAlloc pa(1024*1024*1024);  // 1GB

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    testArgs args = initTestArgs(rank);

    float time = 0;
    const int warmup = 3;
    const int iteration = 5;
    int is_ok = 0;
    TIME_BENCH(time, warmup,
              timeTest(&args, &pa, comm, stream), is_ok);
    TIME_BENCH(time, iteration,
              timeTest(&args, &pa, comm, stream), is_ok);
    std::string ret;
    double delta;
    if (is_ok) {
        ret = check_data(&delta, &args, rank, comm, stream) ? "AC" : "WA";
    } else {
        ret = "WA";
    }
    if (rank == 0) {
        bool is_print_header = true;
        printHeader(is_print_header);
        printOut(args.count, time, ret, delta);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    freeTestArgs(args);
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
