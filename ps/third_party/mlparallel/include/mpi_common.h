/***************************************************************************
 *
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * $Id$
 *
 **************************************************************************/

/**
* @file common.h
* @author wuzhihua02(work@baidu.com)
* @date 2016/1/4 14:36:00
* @version mlparallel-libv_0.0.4.0 + upload model.
* @brief: define for const
*
**/
#ifndef MPIWRAPPER_MPI_COMMON_H
#define MPIWRAPPER_MPI_COMMON_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <float.h>
#include <string>
#include <mpi.h>

namespace mpi_wrapper {
//#define MPIWRAPPER_PARAM_DEBUG
//#define NO_LOG_MPI_DETAIL
//#define NOUSE_OPENMP_FOR_SERVER

#define CHECK_HEARTBEAT
#define CHECK_COMMUNICATION_BLOCKED
#define CHECK_ALL_ABORT
//#define GPU

// print logs
#define ML_LOG(log_out, log_type, ...) {\
    fprintf(log_out, "[%s][%s:%d:%s]: ", #log_type, __FILE__, __LINE__, __FUNCTION__);\
    fprintf(log_out, __VA_ARGS__);\
}

#define ML_INFO(...) ML_LOG(stdout, INFO, __VA_ARGS__)
#define ML_ERROR(...) ML_LOG(stderr, ERROR, __VA_ARGS__)
#define ML_WARNING(...) ML_LOG(stderr, WARNING, __VA_ARGS__)

#ifdef NO_LOG_MPI_DETAIL
#define ML_DEBUG(...)
#else
#define ML_DEBUG(...) ML_LOG(stdout, DEBUG, __VA_ARGS__)
#endif

#define MPI_CHECK(x) if ((x) < 0) {ML_ERROR("Fail to call the function\n"); return -1;}

// support double and real
#ifdef DTYPE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
#else
typedef float real;
#define MPI_REAL_TYPE MPI_FLOAT
#endif

/**
 *  @brief: process type: server-0, worker-1, predictor-2, idle-3.
 */
enum node_type_t {
    NODE_TYPE_SERVER = 0,
    NODE_TYPE_WORKER = 1,
    NODE_TYPE_PREDICTOR = 2,
    NODE_TYPE_IDLE = 3
};

// const variables
const int MPIWRAPPER_OK = 0;
const int MPIWRAPPER_ERROR = -1;
const int MAX_DENSE_NUM = 512;
const int MAX_SPARSE_NUM = 128;
const int MAX_SERVER_NUM = 1000;
const int MAX_STRING_LEN = 2048;
const float DOWNLOAD_PERCENT = 0.9;
const int COMPUTE_THREAD_NUM = 6;
const int CHECK_UPDATE_CONFIG_BATCH = 1;
const float MIN_FLOAT_NUM = FLT_EPSILON;
const int TAG_PUSH_PULL = 0;
const int TAG_NOTICE_FINISH = 2;
const int TAG_NOTICE_RESTORE = 3;
const int TAG_NOTICE_SSP_EPOCH = 4;
const int TAG_NOTICE_ABORT_ALL = 5;
const int MAX_URLS_ONE_QUERY = 2048;
const int BSP_BARRIER_EPOCH_GAP = 1;
const int SSP_BARRIER_EPOCH_GAP = 20;
const int MAX_COMM_MINUTE_GAP = 60;
const int MAX_COMM_TIMES_GAP = 200;
const int MAX_DOWNLOAD_FILE_GAP = 50;
const std::string LOCAL_OUTPUT = "./output/";
const char OTHER_DATA_DELIM = ' ';
}
#endif  // MPIWRAPPER_MPI_COMMON_H

/* vim: set ts=4 sw=4 sts=4 tw=100 */
