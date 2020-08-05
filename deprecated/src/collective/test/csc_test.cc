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

extern float data[CHUNKSIZE];

class CudaPreAlloc{
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

  ~CudaPreAlloc() {
    CUDA_CHECK(cudaFree(_ptr));
  }
};

void warmUpSync(ncclComm_t comm, cudaStream_t stream) {
    float* tmp = static_cast<float*>(AllocateCuda(1024*1024*sizeof(float)));
    for (int i = 0; i < 5; i++) {
      NCCLCHECK(ncclAllReduce(tmp, tmp, 1024*1024, ncclFloat, ncclSum, comm, stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    DeleteCuda(tmp);
}

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
    size_t count;           //0
    size_t chunk_width;     //1
    int chunks;             //2
    int k;                  //3
    float* gradient;        //4
    float* ipt_chunks;      //5
    int* ipt_idx;           //6
    bool* ipt_bitmap;       //7
    float* norms;           //8

    float* gradient_test;   //9,test
    bool* ipt_bitmap_test;  //10,test
} testArgs;

testArgs initTestArgs(int rank) {
  #define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

    // 0 1 2 3
    const size_t count = 1024*1024*16; // float,16*4=64MB
    const size_t chunk_width = 1024*8; // float,8*4=32KB
    const int chunks = DIVUP(count, chunk_width);
    const int k = DIVUP(chunks, 10); // select 10% as important chunks

    // 4 5 6
    float* gradient = static_cast<float*>(AllocateCuda(sizeof(float)*count));
    float* ipt_chunks = static_cast<float*>(AllocateCuda(sizeof(float)*chunk_width*k));
    int* ipt_idx = static_cast<int*>(AllocateCuda(sizeof(int)*k));

    // randomize gradient data
    FeedInputFloat(gradient, count, g_chunk, CHUNKSIZE);

    // 7 8
    bool* ipt_bitmap = static_cast<bool*>(AllocateCuda(sizeof(bool)*chunks));
    float* norms = static_cast<float*>(AllocateCuda(sizeof(float)*chunks));

    // ipt_bitmap should be memset with 0 
    CUDA_CHECK(cudaMemset(ipt_bitmap, 0, sizeof(bool)*chunks));

    // 9 10
    float* gradient_test = static_cast<float*>(AllocateCuda(sizeof(float)*count));
    bool* ipt_bitmap_test = static_cast<bool*>(AllocateCuda(sizeof(bool)*chunks));

    testArgs args = {
        .count           = count          ,
        .chunk_width     = chunk_width    ,
        .chunks          = chunks         ,
        .k               = k              ,
        .gradient        = gradient       ,
        .ipt_chunks      = ipt_chunks     ,
        .ipt_idx         = ipt_idx        ,
        .ipt_bitmap      = ipt_bitmap     ,
        .norms           = norms          ,
        .gradient_test   = gradient_test  ,
        .ipt_bitmap_test = ipt_bitmap_test,
    };
    return args;
}

void prepareCheckData(testArgs* args) {
    CUDA_CHECK(cudaMemcpy(args->gradient_test, args->gradient, sizeof(float)*args->count,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(args->ipt_bitmap_test, args->ipt_bitmap, sizeof(bool)*args->chunks,
                          cudaMemcpyDeviceToDevice));
}

std::vector<int> sort_val_idx(float* v, const int chunks) {
    std::vector<int> idx(chunks);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] > v[i2];});
    //std::sort(v, v + chunks, [](float a, float b) {return a > b;});
    return idx;
}

bool check_ipt_idx(testArgs* args, std::vector<int>& ipt_idx, const int k, int rank, int nranks) {
    // test ipt_idx is equal in all cards
    std::vector<int> ipt_idx_test(k);
    MPICHECK(MPI_Allreduce(ipt_idx.data(), ipt_idx_test.data(), k, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (ipt_idx != ipt_idx_test) {
      LOGERR("ipt_idx is not equal in all cards");
      return false;
    }

    const size_t count = args->count;
    const size_t chunk_width = args->chunk_width;
    const int chunks = args->chunks;
    
    // test ipt_idx is right. Todo. use norm cal bitmap
    float* gradient_test = static_cast<float*>(malloc(sizeof(float)*count));
    bool* bitmap_test = static_cast<bool*>(malloc(sizeof(bool)*chunks));
    float* norms = static_cast<float*>(malloc(sizeof(float)*chunks));
    CUDA_CHECK(cudaMemcpy(gradient_test, args->gradient_test, sizeof(float)*count,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bitmap_test, args->ipt_bitmap_test, sizeof(bool)*chunks,
                          cudaMemcpyDeviceToHost));

    #define FABS(a) ((a!=-INFINITY) ? fabs(a) : a)
    for (int i = 0; i < chunks; ++i) {
      size_t loop_count = std::min(chunk_width, count - i*chunk_width);
      float sum = 0;
      for (int j = 0; j < loop_count; ++j) {
        sum += FABS(gradient_test[i*chunk_width + j]); 
      }
      sum /= loop_count;
      norms[i] = bitmap_test[i] ? sum/nranks : sum;
    }

    MPICHECK(MPI_Allreduce(MPI_IN_PLACE, norms, chunks, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    
    // sort, get topK
    std::vector<int> idx = sort_val_idx(norms, chunks);
    memset(bitmap_test, 0, sizeof(bool)*chunks);

    {
      int i = 0;
      for (i = 0; i < k; ++i) {
        bitmap_test[idx[i]] = true;
      }

      for (i = 0; i < k; ++i) {
        if (!bitmap_test[ipt_idx[i]]) {
          LOGERR("important idx is not equal in i=%d of k=%d", i, k);
          LOGERR("norms[ipt_idx[i]]=%f norms[idx[i]]=%f, may accuracy problem?",
                 norms[ipt_idx[i]], norms[idx[i]]);
          break;
        }
      }
      if (k - i <= 2) return true; 
      else return false;
    }

    return true;
}

bool check_gradient(double* delta, testArgs* args, std::vector<int>& ipt_idx,
                    int rank, ncclComm_t comm, cudaStream_t stream) {
    const size_t count = args->count;
    const size_t chunk_width = args->chunk_width;
    const int chunks = args->chunks;
    const int k = args->k;
    float* ipt_chunks = static_cast<float*>(AllocateCuda(sizeof(float)*chunk_width*k));
    double* dmax = static_cast<double*>(AllocateCuda(sizeof(double))); 
    SetDelta(dmax, -1);

    // gather val to ipt_chunks
    for (int i = 0; i < k; ++i) {
      int idx = ipt_idx[i];
      size_t copy_count = std::min(chunk_width, count - idx*chunk_width);
      CUDA_CHECK(cudaMemcpyAsync(ipt_chunks + i*chunk_width, args->gradient_test + idx*chunk_width,
                     sizeof(float)*copy_count, cudaMemcpyDeviceToDevice, stream));
    }

    NCCLCHECK(ncclAllReduce(ipt_chunks, ipt_chunks, k*chunk_width, ncclFloat, ncclSum, comm, stream));

    // scatter val to gradient
    for (int i = 0; i < k; ++i) {
      int idx = ipt_idx[i];
      size_t copy_count = std::min(chunk_width, count - idx*chunk_width);
      CUDA_CHECK(cudaMemcpyAsync(args->gradient_test + idx*chunk_width, ipt_chunks + i*chunk_width,
                     sizeof(float)*copy_count, cudaMemcpyDeviceToDevice, stream));
    }

    CheckDelta(args->gradient, args->gradient_test, count, dmax, rank, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    *delta = -1;
    CUDA_CHECK(cudaMemcpy(delta, dmax, sizeof(double), cudaMemcpyDeviceToHost));

    DeleteCuda(ipt_chunks);
    DeleteCuda(dmax);
    if (*delta > 0.001 || *delta < 0) return false;
    return true;
}

bool check_data(double* delta, testArgs* args, int rank, int nranks,
                ncclComm_t comm, cudaStream_t stream) {
    const int k = args->k;

    std::vector<int> ipt_idx(k);
    CUDA_CHECK(cudaMemcpy(ipt_idx.data(), args->ipt_idx, sizeof(int)*k, cudaMemcpyDeviceToHost));

    if(!check_ipt_idx(args, ipt_idx, k, rank, nranks)) return false;
    if(!check_gradient(delta, args, ipt_idx, rank, comm, stream)) return false;
    return true;
}

int timeTest(testArgs* args, ncclComm_t comm, cudaStream_t stream) {
    namespace pdgc = paddle::communication::dgc;

    const size_t count       = args->count      ;
    const size_t chunk_width = args->chunk_width;
    const int    chunks      = args->chunks     ;
    const int    k           = args->k          ;
    float*       gradient    = args->gradient   ;
    float*       ipt_chunks  = args->ipt_chunks ;
    int*         ipt_idx     = args->ipt_idx    ;
    bool*        ipt_bitmap  = args->ipt_bitmap ;
    float*       norms       = args->norms      ;

    /// csc process
    /// 1' back cal gradient
    /// 2' gradient allReduce with important chunks that selected in prev iter
    /// 3' select important chunks from gradient
    ///
    /// in this test, it select important chunks first, then do allReduce
    if(!pdgc::calImportant(ipt_idx, k, ipt_bitmap, norms, chunk_width,
                           gradient, count, comm, stream)) return false;
    return pdgc::cscAllReduce(gradient, count, ipt_chunks, chunk_width, ipt_idx, k, comm, stream);
}

void freeTestArgs(testArgs args) {
    DeleteCuda(args.gradient);
    DeleteCuda(args.ipt_chunks);
    DeleteCuda(args.ipt_idx);
    DeleteCuda(args.ipt_bitmap);
    DeleteCuda(args.norms);
}

int runTest(int argc, char *argv[], int rank, int nranks, ncclComm_t comm) {
    namespace pdgc = paddle::communication::dgc;
    // need init before sparseAllGReduce
    pdgc::dynloadNcclLib();

    CudaPreAlloc pa(1024*1024*1024);  // 1GB

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    testArgs args = initTestArgs(rank);
    
    float time = 0;
    const int warmup = 3;
    int iteration = 5;
    int is_ok = 0;
    TIME_BENCH(time, warmup,
        timeTest(&args, comm, stream), is_ok);
    TIME_BENCH(time, iteration,
        timeTest(&args, comm, stream), is_ok);
    std::string ret;
    double delta;
    if (is_ok) {
        // for check
        prepareCheckData(&args);
        timeTest(&args, comm, stream);

        ret = check_data(&delta, &args, rank, nranks, comm, stream) ? "AC" : "WA";
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
