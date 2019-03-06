#ifndef NCCL_WARP_H
#define NCCL_WARP_H

#include <dlfcn.h>
#include "nccl.h"

namespace paddle {
namespace communication {
namespace dgc{

bool warpNcclSymbols(void);

bool warpNcclCommCount(const ncclComm_t comm, int* count);

bool warpNcclCommCuDevice(const ncclComm_t comm, int* device);

bool warpNcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

bool warpNcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

}  // namespace dgc
}  // namespace communication
}  // namespace paddle

#endif
