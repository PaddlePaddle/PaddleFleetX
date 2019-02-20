/***************************************************************************
 *
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * $Id$
 *
 **************************************************************************/

/**
* @file mpi_server.h
* @author wuzhihua02(wuzhihua02@baidu.com)
* @date 2015/1/15 15:50:00
* @version $Revision$
* @brief server process
*
**/
#ifndef MPIWRAPPER_MPI_SERVER_H
#define MPIWRAPPER_MPI_SERVER_H

#include <stdio.h>
#include <omp.h>
#include <vector>
#include "mpi_common.h"
#include "mpi_model_param.h"
#include "mpi_timer.h"
#include "mpi_wrapper.h"
#include "mpi_math.h"
namespace mpi_wrapper {
/**
 * @brief server node to store global parameters
 */
class ServerNode {
public:
    /**
     * @brief constructor for server node.
     * @param [in] server_index : int - server index, from 0
     * @param [in] rank_id      : int - process id in mpi
     * @param [in] total_proc   : int - total process for worker-server
    **/
    ServerNode(int server_index, int rank_id, int worker_num, int total_proc);

    /**
     * @brief destructor for server node
    **/
    ~ServerNode();

    /**
     * @brief initialize server node: allocate space and copy model parameters
     * @param [in] model : const modelparam& : model parameter
     * @return int : 0-success; <0-error
    **/
    int init(const ModelParam& model);

    /**
     * @brief run server in async communication mode.
     * @return int : 0-success; <0-error
    **/
    int run_async();

private:
    /**
     * @brief update model with recv parameters
     * @param [in] recv_model : const modelparam& - received model delta
     * @param [in] source     : int               - receive from which worker id
     * @return int : 0-success; <0-error
    **/
    int update_model(const real* pdata, int source);

    /**
     * @brief print dense parameters for debug
     * @param [in] dense_index : int - dense index
     * @param [in] dest_dense_data : const dense_t* - dense data
     * @param [in] name : const char* - debug tag name
     * @return void
    **/
    void dump_params_dense(int dense_index, const real* pdata, const char* name);

    /**
     * @brief print sparse parameters for debug
     * @param [in] sparse_index : int - sparse index
     * @param [in] src_sparse_data : const sparse_t* - source sparse data
     * @param [in] dest_sparse_data : const sparse_t* - dest sparse data
     * @param [in] sparse_dim : int - dimension for each word's embedding
     * @param [in] name : const char* - debug tag name
     * @return void
    **/
    void dump_params_sparse(int sparse_index, const real* pdata,
                            const sparse_t* dest_sparse_data,
                            int sparse_dim, const char* name);

    /**
     * @brief print sparse parameters for debug
     * @param [in] sparse_index : int - sparse index
     * @param [in] dest_sparse_data : const sparse_t* - dest sparse data
     * @param [in] name : const char* - debug tag name
     * @return void
    **/
    void dump_params_sparse(int sparse_index, const sparse_t* dest_sparse_data, const char* name);

    /**
     * @brief update model by one thread through omp
     * @param [in] sparse_index : int - sparse index
     * @param [in] source : int - receive from which worker id
     * @param [in] src_sparse_data : const sparse_t* - source sparse data
     * @param [in] start_index : int - start from which word
     * @param [in] key_num : int - need to update for how many words
     * @param [in] sparse_dim : int - dimension for each word's embedding
     * @param [in] dest_sparse_data : const sparse_t* - dest sparse data
     * @return int : 0-success; <0-error
    **/
    int server_update_sparse_func(int sparse_index, int source, const real* pdata,
                                  int start_index, int key_num, int sparse_dim,
                                  sparse_t* dest_sparse_data);

    // server index, from 0
    int _server_index;

    // process id in MPI
    int _rankid;

    // worker number
    int _worker_num;

    // active worker number
    int _active_worker_num;

    // server number
    int _server_num;

    // total worker-server process number
    int _total_proc;

    // model parameter
    ModelParam _model;

    // model parameter for send
    ModelParam _send_model;

    // changed key in server for each worker
    std::vector<KeyParam> _changed_key;
};
}
#endif  // MPI_SERVER_H

/* vim: set ts=4 sw=4 sts=4 tw=100 */
