/***************************************************************************
 *
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * $Id$
 *
 **************************************************************************/

/**
* @file mpi_worker.h
* @author wuzhihua02(wuzhihua02@baidu.com)
* @date 2015/1/15 15:50:00
* @version $Revision$
* @brief worker process for communication
**/
#ifndef MPIWRAPPER_MPI_WORKER_H
#define MPIWRAPPER_MPI_WORKER_H

#include <vector>
#include <stdio.h>
#include <queue>
#include "mpi_common.h"
#include "mpi_fidset.h"
#include "mpi_model_param.h"
#include "mpi_timer.h"
#include "mpi_util.h"
#include "mpi_wrapper.h"
namespace mpi_wrapper {
class MPIWrapper;

/**
 * @brief class for worker node
 */
class WorkerNode {
public:
    /**
     * @brief constructor for worker node
     * @param [in] process_id : int - rank id in MPI
     * @param [in] worker_num : int - worker number
     * @param [in] total_proc : int - total process number for worker and server
     * @param [in] thread_num      : trainer's thread num
     **/
    WorkerNode(int process_id, int worker_num, int total_proc,
               int thread_num);

    /**
     * @brief destructor for worker node
     **/
    ~WorkerNode();

    /**
    * @brief initialize memory space for worker process
    * @param [in] model: const ModelParam& - model parameter
    * @return int : 0-success; <0-error
    **/
    int init(const ModelParam& model);

    /**
    * @brief run worker process instance in parallel-train-communication
    * @return int : 0-success; <0-error
    **/
    int run();

    /**
    * @brief store trained wordids for every thread and sparse
    * @param [in] sparse_index: int - sparse index
    * @param [in] thread_index: int - thread index
    * @param [in] word_num     : int - wordid numbers
    * @param [in] word_ids     : int*- wordid array
    * @return int : 0-success; <0-error
    **/
    int add_fidset(int sparse_index, int thread_index, int word_num, int* word_ids);

    /**
    * @brief check whether training is finished or not
    * @return bool : true-finished; fasle-not finished
    **/
    bool get_all_finish();

    /**
    * @brief set training is finished
    * @return void
    **/
    void set_all_finish();

#ifdef CHECK_ALL_ABORT
    /**
    * @brief check whether it's abort or not
    * @return bool : true-abort; fasle-not abort
    **/
    bool get_all_abort();
#endif

#ifdef CHECK_COMMUNICATION_BLOCKED
    /**
    * @brief check whether communication is blocked or not
    * @return bool : true-blocked; fasle-not blocked
    **/
    bool is_communication_blocked();

    /**
    * @brief get current epoch
    * @return int : current epoch index - from 0
    **/
    int get_cur_epoch();

#endif
    /**
    * @brief check whether it's pushing
    * @return bool : true-is pushing; false-not pushing
    **/
    bool get_need_push();

    /**
    * @brief set need to communication
    * @param [in] cur_epoch: int - current epoch
    * @return void
    **/
    void set_need_push(int cur_epoch);

    /**
    * @brief get times for finished communications
    * @return int : push times
    **/
    int get_push_times();

    /**
    * @brief get current iter
    * @return int : current iter
    **/
    int get_cur_iter();

    // TODO - communicate function in serial
    /**
    * @brief communicate in serial
    * @param [in] epoch: int - current epoch
    * @return int : 0-success; <0-error
    **/
    int communicate(int epoch);

private:
    /**
    * @brief communication fuction for thread call
    * @param [in] void*
    * @return void*
    **/
    static void* comm_param_func(void*);

    /**
    * @brief communication function in worker node
    * @return void
    **/
    void comm_param();

    /**
    * @brief print prameters for debug
    * @param [in] model : const ModelParam& - model parameters
    * @return void
    **/
    void dump_params_dense(const ModelParam& model);

    /**
    * @brief print sparse paramethers for debug
    * @param [in] sparse_index: int - sparse index
    * @param [in] wordid      : const std::vector<int>& - word id vector
    * @param [in] model       : const ModelParam& - model parameters
    * @return int : 0-success; <0-error
    **/
    void dump_params_sparse(int sparse_index,
                            const std::vector<int>& wordid,
                            const ModelParam& model);

    // rank id in MPI
    int _rankid;

    // worker number
    int _worker_num;

    // total process number
    int _total_proc;

    // total thread number in trainers (for trained wordids store)
    int _thread_num;

    // push all models to hadoop after training finished
    int _once_push;

    // communicate once after how many training examples learned
    int _comm_batch;

    // hadoop output path for model
    std::string _hdfs_output;

    // config file name
    std::string _parallel_config_name;

    // final model file path
    std::string _final_modelfile;

    // sparse number
    int _sparse_num;

    // all training is finished or not
    volatile bool _g_all_finish;

    // need to communicate or not
    volatile bool _g_need_push;

    // abort or not
    volatile bool _g_all_abort;

#ifdef CHECK_COMMUNICATION_BLOCKED
    // the time of last communication
    struct timeval _g_last_comm_time;
#endif

    // total communication times
    uint32_t _g_push_times;

    // current epoch
    int _g_cur_epoch;

    // last saved epoch
    int _g_last_epoch;

    // current iter
    int _g_cur_iter;

    // TODO
    // last learned number in communication
    uint64_t _g_last_learnnum_in_comm;

    // last epoch for abort checking
    int _g_last_checkabort_epoch;

    // fid set to store all updated wordids for each sparse and thread
    FidSetParam _g_fidset;

    // model for send
    ModelParam _push_model;

    // model for recv
    ModelParam _pull_model;

    // pointer for MPIWrapper instance
    MPIWrapper* _mpiwrapper_ptr;
};
}
#endif  // MPI_WORKER_H

/* vim: set ts=4 sw=4 sts=4 tw=100 */
