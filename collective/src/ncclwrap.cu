#include "ncclwrap.h"
#include "common.h"
#include <pthread.h>
#include <stdio.h>

namespace paddle {
namespace communication {
namespace dgc{

static enum { ncclUninitialized, ncclInitializing, ncclInitialized, ncclError } ncclState = ncclUninitialized;

/*Function Pointers*/
static const char*  (*ncclGetErrorStringFuncPoint)(ncclResult_t result);
static ncclResult_t (*ncclCommCountFuncPoint)(const ncclComm_t comm, int* count);
static ncclResult_t (*ncclCommCuDeviceFuncPoint)(const ncclComm_t comm, int* device);
static ncclResult_t (*ncclAllReduceFuncPoint)(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
static ncclResult_t (*ncclAllGatherFuncPoint)(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

bool warpNcclSymbols(void) {
  if (ncclState == ncclInitialized) {
    return true;
  } else if (ncclState == ncclError) {
    return false;
  }

  if (__sync_bool_compare_and_swap(&ncclState, ncclUninitialized, ncclInitializing) == false) {
    // Another thread raced in front of us. Wait for it to be done.
    while (ncclState == ncclInitializing) pthread_yield();
    return (ncclState == ncclInitialized) ? true : false;
  }

  static void* ncclhandle = NULL;
  void* tmp = NULL;
  void** cast = NULL;

  ncclhandle=dlopen("libnccl.so", RTLD_NOW);
  if (!ncclhandle) {
    ncclhandle=dlopen("libnccl.so.2", RTLD_NOW);
    if (!ncclhandle) {
      LOGERR("Failed to open libnccl.so[.2]");
      goto teardown;
    }
  }

#define LOAD_SYM(handle, symbol, funcptr) do {                 \
    cast = (void**)&funcptr;                                   \
    tmp = dlsym(handle, symbol);                               \
    if (tmp == NULL) {                                         \
      LOGERR("dlsym failed on %s - %s\n", symbol, dlerror());  \
      goto teardown;                                           \
    }                                                          \
    *cast = tmp;                                               \
  } while (0)

  LOAD_SYM(ncclhandle, "ncclGetErrorString", ncclGetErrorStringFuncPoint);
  LOAD_SYM(ncclhandle, "ncclCommCount", ncclCommCountFuncPoint);
  LOAD_SYM(ncclhandle, "ncclCommCuDevice", ncclCommCuDeviceFuncPoint);
  LOAD_SYM(ncclhandle, "ncclAllReduce", ncclAllReduceFuncPoint);
  LOAD_SYM(ncclhandle, "ncclAllGather", ncclAllGatherFuncPoint);

  ncclState = ncclInitialized;
  return true;

teardown:
  ncclGetErrorStringFuncPoint = NULL;
  ncclCommCountFuncPoint = NULL;
  ncclCommCuDeviceFuncPoint = NULL;
  ncclAllReduceFuncPoint = NULL;
  ncclAllGatherFuncPoint = NULL;

  if (ncclhandle != NULL) dlclose(ncclhandle);
  ncclState = ncclError;
  return false;
}

#define NCCL_CHECK(cmd) do {                        \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    LOGERR("Failed, NCCL error '%s'",               \
           ncclGetErrorStringFuncPoint(r));         \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define FP_CHECK(fp) do {                           \
  if (fp == NULL) {                                 \
    LOGERR("lib nccl not initialized.");            \
    exit(EXIT_FAILURE);                             \
    return false;                                   \
  }                                                 \
} while(0)                                          \
 
bool warpNcclCommCount(const ncclComm_t comm, int* count) {
  FP_CHECK(ncclCommCountFuncPoint);
  NCCL_CHECK(ncclCommCountFuncPoint(comm, count));
  return true;
}

bool warpNcclCommCuDevice(const ncclComm_t comm, int* device) {
  FP_CHECK(ncclCommCuDeviceFuncPoint );
  NCCL_CHECK(ncclCommCuDeviceFuncPoint(comm, device));
  return true;
}

bool warpNcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  FP_CHECK(ncclAllReduceFuncPoint);
  NCCL_CHECK(ncclAllReduceFuncPoint(sendbuff, recvbuff, count, datatype, op, comm, stream));
  return true;
}

bool warpNcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  FP_CHECK(ncclAllGatherFuncPoint);
  NCCL_CHECK(ncclAllGatherFuncPoint(sendbuff, recvbuff, sendcount, datatype, comm, stream));
  return true;
}

}  // namespace dgc
}  // namespace communication
}  // namespace paddle
