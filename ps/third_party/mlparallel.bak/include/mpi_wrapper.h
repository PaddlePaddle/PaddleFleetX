/***************************************************************************
 *
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * $Id$
 *
 **************************************************************************/

/**
* @file mpi_wrapper.h
* @author wuzhihua02(wuzhihua02@baidu.com)
* @date 2016/01/15 15:50:00
* @version $Revision$
* @brief mpi wrapper for overall MPI environment and API
*
**/
#ifndef MPIWRAPPER_MPI_WRAPPER_H
#define MPIWRAPPER_MPI_WRAPPER_H

#include <malloc.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include "mpi_acevaluator.h"
#include "mpi_common.h"
#include "mpi_model_param.h"
#include "mpi_server.h"
#include "mpi_timer.h"
#include "mpi_worker.h"
#include "mpi_math.h"
namespace mpi_wrapper {
class WorkerNode;

/**
 * @brief class for MPIWrapper
 */
class MPIWrapper {
public:
    /**
    * @brief get Singleton instance of MPIWrapper for one process
    * @return static MPIWrapper& : MPIWrapper object
    **/
    static MPIWrapper& instance();

    /**
    * @brief destructor for MPIWrapper
    **/
    ~MPIWrapper();

    /**
    * @brief get current process' id
    * @return int : rank id of the current process
    **/
    int get_rankid();

    /**
    * @brief check whether it's server or not
    * @return bool : true-yes; false-no
    **/
    bool is_server();

    /**
    * @brief check whether a process is a worker or not.
    * @return bool : true-yes; false-no
    **/
    bool is_worker();

    /**
    * @brief check whether a process is a predictor or not
    * @return bool : true-yes; false-no
    **/
    bool is_predictor();

    /**
    * @brief check whether a process is a idle node or not.
    * @param [in] model: const ModelParam& - model parameter
    * @return bool : true-yes; false-no
    **/
    bool is_idler();

    /**
    * @brief init mpi environment.
    * @param [in] nodes: total node number
    * @param [in] proc_per_node: proc per node
    * @return int : 0-success; <0-error
    **/
    int init(int nodes, int proc_per_node);

    /**
    * @brief start server to receive, update model and send parameters to workers.
    * @param [in] model_param: const ModelParam& - model parameter
    * @return int : 0-success; <0-error
    **/
    int start_server(ModelParam& model_param);

    /**
    * @brief start worker for communication with servers
    * @param [in] model_param: const ModelParam& - model parameter
    * @param [in] thread_num: trainer's training thread num
    **/
    int start_worker(ModelParam& model_param, int thread_num);

    /**
    * @brief call MPI_Abort to terminate all processes
    * @return void
    **/
    void abort_all();

    /**
    * @brief barrier all workers
    * @return int : 0-success; <0-error
    **/
    int barrier_worker();

    /**
    * @brief barrier all servers and workers
    * @return int : 0-success; <0-error
    **/
    int barrier_server_worker();

    /**
    * @brief barrier all processes
    * @return int : 0-success; <0-error
    **/
    int barrier_allproc();

    /**
    * @brief push_and_pull for workers
    * @param [in] send_model: ModelParam& - model parameter for send
    * @param [out] recv_model: ModelParam& - model buffer for receive
    * @return int : 0-success; <0-error
    **/
    int push_and_pull(ModelParam& send_model,
                      ModelParam& recv_model);

    /**
    * @brief server receives any source and tag data form workers' push
    * @param [out] recv_model: ModelParam& - model buffer for receive
    * @param [out] source    : int& - which worker received from
    * @param [out] tag       : int& - which tag received from
    * @return int : 0-success; <0-error
    **/
    int server_recv(real* &pdata, int& source, int& tag);

    /**
    * @brief servers send data to workers' pull
    * @param [in] source : - which worker to send to
    * @param [in] model  : const ModelParam& - model parameter
    * @return int : 0-success; <0-error
    **/
    int server_send(int source, const ModelParam& model);

    //TODO: split big message into chunks
    /**
    * @brief call MPI_Recv to receive push data from any source and tag
    * @param [out] pdata: real* - receive buffer
    * @param [out] recv_len: int& - receive len
    * @param [out] source    : int& - which worker received from
    * @param [out] tag       : int& - which tag received from
    * @return int : 0-success; <0-error
    **/
    int recv_data(real* pdata, int& recv_len, int& source, int& tag);

    //TODO: split big message into chunks
    /**
    * @brief call MPI_Send to send data.
    * @param [in] pdata: real* - send buffer
    * @param [in] len: int - send len
    * @param [in] dest: int - dest process' id
    * @param [in] tag: int - data tag: TAG_*
    * @return int : 0-success; <0-error
    **/
    int send_data(real* pdata, int len, int dest, int tag);

    /**
    * @brief get server index
    * @return int: server index
    **/
    int get_server_index();

    /**
    * @brief get worker index.
    * @return int : worker index
    **/
    int get_worker_index();

    /**
    * @brief check whether the current process is the first worker or not
    * @return bool : true-yes; false-no
    **/
    bool is_first_worker();

    /**
    * @brief check whether the current process is the last worker or not
    * @return bool : true-yes; false-no
    **/
    bool is_last_worker();

    /**
    * @brief get the server index in one machine's all servers
    * @return int : server index in one machine
    **/
    int get_server_index_one_machine();

    /**
    * @brief get worker index in one machine
    * @return int : worker index in one machine
    **/
    int get_worker_index_one_machine();

    /**
    * @brief get the current process' worker index or server index in one machine
    * @return int : worker or server index in one machine
    **/
    int get_worker_or_server_index_one_machine();

    /**
    * @brief check whether the current process is the first process in one machine
    * @return bool : true-yes; false-no
    **/
    bool is_first_proc_one_machine();

    /**
    * @brief get worker number
    * @return int : worker number
    **/
    int get_worker_num();

    /**
    * @brief get worker-server mode
    * @return int : 0-server0 server1 worker0...; 1-server0 worker0 ...
    **/
    int get_server_mode();

    /**
    * @brief get server rankid in MPI by server index
    * @param [in] index: int - server index from 0
    * @return int : server process id in MPI
    **/
    int get_server_rankid(int index);

    /**
    * @brief get worker rankid in MPI by worker index
    * @param [in] index: int - worker index from 0
    * @return int : worker process id in MPI
    **/
    int get_worker_rankid(int index);

    /**
    * @brief worker notice server to finish
    * @return int : 0-success; <0-error
    **/
    int notice_finish();

    /**
    * @brief get worker node pointer
    * @return WorkerNode* - worker node pointer
    **/
    WorkerNode* get_worker_node();

    /**
    * @brief divide dense data into small parts for each server in splitmode1
    * @param [in] server_index: int - server index
    * @param [in] total_row   : int - total row number
    * @param [in] col         : int - column number
    * @param [out] this_server_start_pos : int - this server's start pos
    * @param [out] this_server_row : int - this server's row number
    * @return int : 0-success; <0-error
    **/
    int divide_densedata_for_each_node_mode1(int server_index,
            int total_row,
            int col,
            int& this_server_start_pos,
            int& this_server_row);

    /**
    * @brief divide dense data into small parts for each server in splitmode0
    * @param [in] server_index: int - server index
    * @param [in] total_row   : int - total row number
    * @param [in] col         : int - column number
    * @param [out] this_server_start_pos : int - this server's start pos
    * @param [out] this_server_row : int - this server's row number
    * @return int : 0-success; <0-error
    **/
    int divide_densedata_for_each_node_mode0(int server_index,
            int total_row,
            int col,
            int& this_server_start_pos,
            int& this_server_row);

    /**
    * @brief initilize dense data part space for each server
    * @param [in] model: const ModelParam& - model parameter
    * @return int : 0-success; <0-error
    **/
    //int init_server_dense_part(const ModelParam& model);

    /**
    * @brief free dense data part space for each server
    * @return void
    **/
    //void free_server_dense_part();

    /**
    * @brief get server dense part by server index
    * @param [in] server_index : int - server index
    * @return const server_dense_part_t* : server dense part structure
    **/
    //const server_dense_part_t* get_server_dense_parts(int server_index);

    /**
    * @brief get server dense part number by server idnex
    * @param [in] server_index : int - server index
    * @return int : dense part number
    **/
    //int get_server_dense_part_num(int server_index);

    /**
    * @brief get dense part for one server index and parse index
    * @param [in] server_index : int - server index
    * @param [in] part_index : int - part index
    * @return const dense_part_t* - dense part for given sparse and part index
    **/
    const dense_part_t* get_dense_part(int server_index, int part_index);

#ifdef CHECK_HEARTBEAT
    /**
    * @brief check whether any workers fail to heartbeat
    * @param [in] active_worker_num : int - active worker number
    * @return bool: true-fail; false-no fail
    **/
    bool is_heartbeat_failed(int active_worker_num);
#endif

private:
    /**
    * @brief constructor for MPIWrapper
    **/
    MPIWrapper();

    /**
    * @brief copy constructor for MPIWrapper
    * @param [in] other : const MPIWrapper& - other model
    **/
    MPIWrapper(const MPIWrapper& other);

    /**
    * @brief assign function
    * @param [in] other : const MPIWrapper& - other model
    * @return MPIWrapper& : MPIWrapper object
    **/
    MPIWrapper& operator=(const MPIWrapper& other);

    /**
    * @brief allocate memory for send and recv buffer
    * @param [in] model_size : int - model size
    * @return int : 0-success; <0-error
    **/
    int init_buffer(ModelParam& model_param, int decount_num);

    /**
    * @brief set server ids information
    * @return int : 0-success; <0-error
    **/
    int set_serverinfo();

    /**
    * @brief send model to server index
    * @param [in] server_index : int - server index
    * @param [in] send_model   : ModelParam& - model for send
    * @return int : 0-success; <0-error
    **/
    int push(int server_index, ModelParam& send_model, int tag);

    /**
    * @brief receive from given server index
    * @param [in] server_index : int - server index
    *  @param [out] recv_model  : ModelParam& - model buffer for recv
    * @return int : 0-success; <0-error
    **/
    int pull(int server_index, ModelParam& recv_model);

    //TODO: split big message into chunks
    /**
    * @brief call MPI_Recv to receive specific data from specific source
    * @param [in] pdata : real* - buffer for receive
    * @param [out] len   : int&  - receive length
    * @param [in] source : int - source process id
    * @param [in] tag : int - data tag
    * @return int : 0-success; <0-error
    **/
    int recv_specific_data(real* pdata, int& len, int source, int tag);

    /**
    * @brief update send_model through data from recv_model
    * @param [out] send_model : ModelParam& - model for update
    * @param [in] recv_model   : const ModelParam& - recv model
    * @return int : 0-success; <0-error
    **/
    int update_model(ModelParam& send_model, const ModelParam& recv_model);

    /**
    * @brief print dense parameters for debug
    * @param [in] dense_index : int - dense index
    * @param [in] src_dense_data : const dense_t* - source dense data
    * @param [in] dest_dense_data: const dense_t* - dest dense data
    * @return int : 0-success; <0-error
    **/
    void dump_params_dense(int dense_index,
                           const dense_t* src_dense_data,
                           const dense_t* dest_dense_data);

    /**
    * @brief print sparse parameters for debug
    * @param [in] key_num : int - key number
    * @param [in] sparse_index : int - sparse index
    * @param [in] src_sparse_data : const sparse_t* - source sparse data
    * @param [in] dest_sparse_data: const sparse_t* - dest sparse data
    * @param [in] sparse_dim : int - sparse dimension for one word
    * @return int : 0-success; <0-error
    **/
    void dump_params_sparse(int key_num,
                            int sparse_index,
                            const sparse_t* src_sparse_data,
                            const sparse_t* dest_sparse_data,
                            int sparse_dim);

    /**
    * @brief print sparse parameters for debug
    * @param [in] key_num : int - key number
    * @param [in] pkey : const int* - word ids array
    * @param [in] pdata: const real* - pointer for model data
    * @param [in] sparse_dim : int - sparse dimension for one word
    * @param [in] name : const char* - debug data tag name
    * @return int : 0-success; <0-error
    **/
    void dump_params_sparse(int key_num,
                            const int* pkey,
                            const real* pdata,
                            int sparse_dim,
                            const char* name);

    /**
    * @brief get node type
    * @return int : 0-server 1-worker 2-perdictor 3-idle
    **/
    int get_nodetype();

    // process id in MPI rank
    int _rankid;

    // total process number, include predictor
    int _total_proc_num;

    // server_num and worker_num
    int _total_server_worker;

    // send buffer
    real* _comm_buf;

    int _server_mode;

    // server number
    int _server_num;

    // worker number
    int _worker_num;

    // predictor number
    int _predictor_num;

    // total node num
    int _nodes;

    // process per node/machine (=2)
    int _proc_per_node;

    // process per node/machine in actual
    int _proc_per_node_actual;

    // <server index, server process id>
    int _server_ids[MAX_SERVER_NUM];

    // server MPI COMM
    MPI_Comm _server_comm;

    // <worker index, worker process id>
    int _worker_ids[MAX_SERVER_NUM];

    // <worker id, worker epoch>
    int _worker_epoch_infos[MAX_SERVER_NUM];

#ifdef CHECK_HEARTBEAT
    // <worker id, last comm time>
    struct timeval _last_comm_time[MAX_SERVER_NUM];
#endif

    // worker MPI COMM
    MPI_Comm _worker_comm;

    // predictor MPI COMM
    MPI_Comm _predictor_comm;

    // predictor and idle MPI COMM
    MPI_Comm _predict_and_idle_comm;

    // server and worker MPI COMM
    MPI_Comm _server_and_worker_comm;

    // worker node pointer
    WorkerNode* _worker_node;

    // data part for servers
    server_dense_part_t* _server_dense_part;

    // send and receive buffer size in byte
    uint64_t _max_buffer_size;

    // send and receive buffer size in int
    uint64_t _max_recv_buf_count;

    // dense split mode
    int _dense_comm_divide_mode;

    /**
    * @brief call back functions to call before update_model() function
    * @return void
    **/
    void (*_call_back_before_update)();

    /**
    * @brief call back functions to call after update_model() function
    * @return void
    **/
    void (*_call_back_after_update)();
};
}

#endif  // MPIWRAPPER_H

/* vim: set ts=4 sw=4 sts=4 tw=100 */
